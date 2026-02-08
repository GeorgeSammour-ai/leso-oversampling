# leso_csv_tool.py
# Generic LESO oversampling tool: CSV in → oversampled CSV out
# - Prompts user for: label column, minority/majority labels, target minority % of majority
# - Optional dropping of ID-like columns
# - Keeps numeric columns numeric and categorical columns categorical in output
#   (by decoding synthetic one-hot blocks back to the most likely category via argmax)

import os
import sys
import argparse
import re  # FIX: needed for ID-column suggestions
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EPS = 1e-12


def binary_entropy(p, eps=EPS):
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _rng(seed: int):
    return np.random.default_rng(seed)


@dataclass
class LESO:
    """
    Latent Entropy-guided Synthetic Oversampling (LESO)

    Works on:
      - X: numeric feature matrix (numpy array)
      - y: binary labels with minority=1, majority=0

    Safety rule:
      - If minority count < 3, returns data unchanged.
    """
    n_states: int = 4
    k_neighbors: int = 5
    alpha: float = 1.0
    c_beta: float = 2.0

    def _global_fallback(self, X, y, G, rng):
        idx_min = np.where(y == 1)[0]
        if idx_min.size < 3:
            return []

        X_min = X[idx_min]
        k_eff = min(self.k_neighbors, idx_min.size - 1)
        if k_eff < 1:
            return []

        nn_min = NearestNeighbors(n_neighbors=k_eff + 1).fit(X_min)
        neigh_min = nn_min.kneighbors(X_min, return_distance=False)[:, 1:]

        k_local = min(self.k_neighbors, X.shape[0] - 1)
        nn_all = NearestNeighbors(n_neighbors=k_local + 1).fit(X)
        neigh_all = nn_all.kneighbors(X_min, return_distance=False)[:, 1:]
        p_i = np.array([np.mean(y[nn_ids] == 1) for nn_ids in neigh_all])
        H_i = binary_entropy(p_i)

        probs = (H_i / H_i.sum()) if H_i.sum() > 0 else np.ones_like(H_i) / len(H_i)
        chosen = rng.choice(len(idx_min), size=G, replace=True, p=probs)

        X_new = []
        for ii in chosen:
            x_i = X_min[ii]
            nn_idx = rng.choice(neigh_min[ii])
            x_nn = X_min[nn_idx]

            h = float(H_i[ii])
            a = b = 1.0 + self.c_beta * (1.0 - h)
            lam = rng.beta(a, b)
            X_new.append(x_i + lam * (x_nn - x_i))
        return X_new

    def fit_resample(self, X, y, n_to_add: int, random_state=0):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        rng = _rng(random_state)

        n_min = int(np.sum(y == 1))
        if n_min < 3 or n_to_add <= 0:
            return X, y

        G = int(n_to_add)

        # Latent-state inference via GMM (robust for tabular numeric data)
        S = min(self.n_states, max(2, min(6, X.shape[0] // 30 + 2)))
        gmm = GaussianMixture(
            n_components=S,
            covariance_type="full",
            random_state=random_state,
            reg_covar=1e-6
        )
        z = gmm.fit_predict(X)

        state_weights = np.zeros(S, dtype=float)
        state_min_counts = np.zeros(S, dtype=int)

        for s in range(S):
            idx = np.where(z == s)[0]
            if idx.size == 0:
                continue
            y_s = y[idx]
            pi_s = float(np.mean(y_s == 1))
            H_s = binary_entropy(pi_s)
            # weight: entropy^alpha times "need" (1 - pi_s)
            state_weights[s] = (H_s ** self.alpha) * (1.0 - pi_s)
            state_min_counts[s] = int(np.sum(y_s == 1))

        viable = state_min_counts >= 3

        # If no viable states, fallback to global
        if not viable.any():
            X_new_list = self._global_fallback(X, y, G, rng)
            if not X_new_list:
                return X, y
            X_new = np.vstack(X_new_list)
            y_new = np.ones(X_new.shape[0], dtype=int)
            return np.vstack([X, X_new]), np.concatenate([y, y_new])

        w = state_weights.copy()
        w[~viable] = 0.0
        if w.sum() <= 0:
            w[viable] = 1.0
        alloc = w / w.sum()

        # Allocate samples per state
        G_s = np.floor(G * alloc).astype(int)
        remainder = G - G_s.sum()
        if remainder > 0:
            order = np.argsort(-alloc)
            for i in range(remainder):
                G_s[order[i % len(order)]] += 1

        X_new_list = []

        for s in range(S):
            if G_s[s] <= 0:
                continue

            idx = np.where(z == s)[0]
            if idx.size == 0:
                continue

            idx_min = idx[y[idx] == 1]
            if idx_min.size < 3:
                continue

            X_min = X[idx_min]

            # neighbors within minority subset (for interpolation)
            k_eff = min(self.k_neighbors, idx_min.size - 1)
            if k_eff < 1:
                continue

            nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(X_min)
            neigh = nn.kneighbors(X_min, return_distance=False)[:, 1:]

            # local entropy within state neighborhood
            X_state = X[idx]
            y_state = y[idx]
            k_local = min(self.k_neighbors, len(idx) - 1)
            nn_state = NearestNeighbors(n_neighbors=k_local + 1).fit(X_state)
            state_neigh = nn_state.kneighbors(X_min, return_distance=False)[:, 1:]

            p_i = np.array([np.mean(y_state[nn_ids] == 1) for nn_ids in state_neigh])
            H_i = binary_entropy(p_i)

            probs = (H_i / H_i.sum()) if H_i.sum() > 0 else np.ones_like(H_i) / len(H_i)
            chosen = rng.choice(len(idx_min), size=G_s[s], replace=True, p=probs)

            # Generate synthetic samples
            for ii in chosen:
                x_i = X_min[ii]
                nn_idx = rng.choice(neigh[ii])
                x_nn = X_min[nn_idx]

                h = float(H_i[ii])
                a = b = 1.0 + self.c_beta * (1.0 - h)
                lam = rng.beta(a, b)
                X_new_list.append(x_i + lam * (x_nn - x_i))

        # fallback if statewise didn't generate
        if not X_new_list:
            X_new_list = self._global_fallback(X, y, G, rng)

        if not X_new_list:
            return X, y

        X_new = np.vstack(X_new_list)
        y_new = np.ones(X_new.shape[0], dtype=int)
        return np.vstack([X, X_new]), np.concatenate([y, y_new])


def build_preprocessor(X_df: pd.DataFrame):
    # Identify numeric vs categorical columns from pandas dtypes
    num_cols = X_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X_df.columns if c not in num_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    if not transformers:
        raise ValueError("No feature columns found.")

    pre = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
    return pre


def decode_back(pre: ColumnTransformer, X_res: np.ndarray) -> pd.DataFrame:
    """
    Convert numeric working-space matrix back to original schema:
      - Numeric: inverse StandardScaler
      - Categorical: decode one-hot blocks by argmax within each categorical column block
    """
    out_parts = {}
    col_start = 0

    for name, trans, cols in pre.transformers_:
        if name == "num":
            scaler = trans.named_steps["scaler"]
            n_num = len(cols)
            X_num_scaled = X_res[:, col_start:col_start + n_num]
            X_num = scaler.inverse_transform(X_num_scaled)
            for j, c in enumerate(cols):
                out_parts[c] = X_num[:, j]
            col_start += n_num

        elif name == "cat":
            ohe = trans.named_steps["onehot"]
            cats = ohe.categories_
            for c, cat_values in zip(cols, cats):
                k = len(cat_values)
                X_cat_block = X_res[:, col_start:col_start + k]
                idx = np.argmax(X_cat_block, axis=1)
                decoded = np.array([cat_values[i] for i in idx], dtype=object)
                out_parts[c] = decoded
                col_start += k

    return pd.DataFrame(out_parts)


def prompt_choice(prompt, options):
    print(prompt)
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt}")
    while True:
        s = input("Enter choice number: ").strip()
        if s.isdigit():
            k = int(s)
            if 1 <= k <= len(options):
                return options[k - 1]
        print("Invalid choice. Try again.")


def prompt_yes_no(prompt, default_yes=True):
    d = "Y/n" if default_yes else "y/N"
    while True:
        s = input(f"{prompt} ({d}): ").strip().lower()
        if s == "" and default_yes:
            return True
        if s == "" and not default_yes:
            return False
        if s in ["y", "yes"]:
            return True
        if s in ["n", "no"]:
            return False
        print("Please enter y or n.")


def main():
    parser = argparse.ArgumentParser(
        description="LESO oversampling tool: CSV in → oversampled CSV out (generic)"
    )
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", default=None, help="Path to output CSV (default: <input>_LESO.csv)")
    parser.add_argument("--seed", type=int, default=11, help="Random seed")
    args = parser.parse_args()

    in_path = args.input.strip().strip('"').strip("'")
    if not os.path.exists(in_path):
        print("File not found:", in_path)
        sys.exit(1)

    df = pd.read_csv(in_path)
    print("\nLoaded:", in_path)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # pick label column
    class_col = prompt_choice("\nSelect the class (label) column:", list(df.columns))

    feature_cols = [c for c in df.columns if c != class_col]

    # optional: drop ID-like columns
    if prompt_yes_no("\nDo you want to drop ID-like columns (recommended)?", default_yes=True):
        # heuristic suggestions for ID-like columns
        suggestions = [
            c for c in feature_cols
            if re.search(r"(id$|_id$|^id$|passengerid$)", c.lower())
        ]
        if suggestions:
            print("\nSuggested ID columns:", suggestions)

        drop_cols = []
        while True:
            s = input("Enter columns to drop separated by commas (blank = none): ").strip()
            if s == "":
                break
            cand = [x.strip() for x in s.split(",") if x.strip()]
            bad = [x for x in cand if x not in feature_cols]
            if bad:
                print("Not found:", bad)
                continue
            drop_cols = cand
            break

        if drop_cols:
            df = df.drop(columns=drop_cols)
            print("Dropped:", drop_cols)

    # label counts
    vc = df[class_col].astype(str).value_counts(dropna=False)
    print("\nObserved labels and counts:")
    for lab, cnt in vc.items():
        print(f" - {lab}: {cnt}")

    minority_label = input("\nEnter the MINORITY label exactly as shown: ").strip()
    majority_label = input("Enter the MAJORITY label exactly as shown: ").strip()

    uniq = set(df[class_col].astype(str).unique())
    if minority_label not in uniq or majority_label not in uniq:
        print("One (or both) labels not found in data. Re-run and enter exactly.")
        sys.exit(1)

    # target minority as % of majority
    while True:
        s = input("\nTarget minority as % of majority (100=balance, 50=half, 200=twice): ").strip()
        try:
            target_pct = float(s)
            if target_pct >= 0:
                break
        except ValueError:
            pass
        print("Invalid number. Try again.")

    # output path
    if args.output:
        out_path = args.output
    else:
        base, ext = os.path.splitext(in_path)
        out_path = base + "_LESO.csv"

    # Map y to {0,1}
    y_str = df[class_col].astype(str)
    y = np.where(
        y_str == minority_label, 1,
        np.where(y_str == majority_label, 0, np.nan)
    )

    if np.isnan(y).any():
        print("\nBinary-only tool: class column contains labels beyond the two specified.")
        print("Please filter your dataset to two classes or extend the tool for multiclass.")
        sys.exit(1)

    y = y.astype(int)
    n_min = int((y == 1).sum())
    n_maj = int((y == 0).sum())
    print(f"\nBEFORE: minority={n_min}, majority={n_maj}, IR={n_maj/max(n_min,1):.3f}")

    target_min = int(round((target_pct / 100.0) * n_maj))
    n_to_add = max(0, target_min - n_min)

    print(f"Target minority count: {target_min} ({target_pct:.1f}% of majority)")
    print(f"Will add: {n_to_add} synthetic minority samples")

    # preprocess features and run LESO
    X_df = df.drop(columns=[class_col])
    pre = build_preprocessor(X_df)
    X_work = pre.fit_transform(X_df)

    leso = LESO(n_states=4, k_neighbors=5, alpha=1.0, c_beta=2.0)
    X_res, y_res = leso.fit_resample(X_work, y, n_to_add=n_to_add, random_state=args.seed)

    # decode back to original schema
    X_back = decode_back(pre, X_res)
    y_back = np.where(y_res == 1, minority_label, majority_label)

    out_df = X_back.copy()
    out_df[class_col] = y_back
    out_df.to_csv(out_path, index=False)

    n_min2 = int((y_res == 1).sum())
    n_maj2 = int((y_res == 0).sum())
    print(f"\nAFTER : minority={n_min2}, majority={n_maj2}, IR={n_maj2/max(n_min2,1):.3f}")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
