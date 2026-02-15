# THIS IS THE OFFICIAL IMPLEMENTATION USED IN THE IJMLC SUBMITTED PAPER.
# The manuscript experiments were produced using:
#   LEO_Submitted (mode="submitted")
#
# The class LEO_MinorityPosterior corresponds to an alternative research
# variant and is NOT the one used for the main experimental results.
# leo_csv_tool.py
# Generic LEO oversampling tool: CSV in → oversampled CSV out
#
# This script supports TWO LEO variants because you have (at least) two paper drafts:
#
# (A) "submitted" (matches: Synthetic Oversampling LEO Submitted Version.docx)
#     - Fit GMM on the full training feature space X (all classes)
#     - Component minority proportion p_k computed from y within each component
#     - Component entropy H_k = binary_entropy(p_k) (class-mixture overlap)
#     - Allocate synthetic samples with w_k = (H_k^alpha) * (1 - p_k)
#     - Within each component, compute instance-level entropy for minority anchors:
#         p_i = proportion of minority labels among kNN (within the same component)
#         H_i = binary_entropy(p_i)
#     - Generate synthetic samples by interpolation between minority neighbors within the same component
#       using lambda ~ Uniform(0,1)
#
# (B) "minority_posterior" (matches: Synthetic Oversampling LEO (1).docx)
#     - Fit GMM on minority space X_min
#     - Use posterior-membership entropy over responsibilities gamma as component uncertainty
#     - Use similarity-entropy within minority neighborhoods for instance-level uncertainty
#     - Uses lambda ~ Beta(a,b) modulated by instance entropy
#
# Choose with: --mode submitted  OR  --mode minority_posterior
# Default is --mode submitted to align with the journal-submitted manuscript.
#
# Notes:
# - This tool is binary-label (minority vs majority). For multiclass, filter to 2 classes first.
# - Oversampling MUST be applied to the training split only inside CV to avoid leakage.

import os
import sys
import argparse
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EPS = 1e-12


def safe_log(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    return np.log(np.clip(x, eps, None))


def binary_entropy(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def _rng(seed: int):
    return np.random.default_rng(int(seed))


def build_preprocessor(X_df: pd.DataFrame) -> ColumnTransformer:
    num_cols = X_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X_df.columns if c not in num_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    if not transformers:
        raise ValueError("No feature columns found.")

    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def decode_back(pre: ColumnTransformer, X_res: np.ndarray) -> pd.DataFrame:
    out_parts: Dict[str, Any] = {}
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


def prompt_choice(prompt: str, options: List[str]) -> str:
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


def prompt_yes_no(prompt: str, default_yes: bool = True) -> bool:
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


@dataclass
class LEO_Submitted:
    n_components: int = 4
    k_neighbors: int = 5
    alpha: float = 1.0
    reg_covar: float = 1e-6

    def fit_resample(self, X: np.ndarray, y: np.ndarray, n_to_add: int, random_state: int = 11) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        rng = _rng(random_state)

        n_min = int(np.sum(y == 1))
        n_maj = int(np.sum(y == 0))
        diag: Dict[str, Any] = {
            "mode": "submitted",
            "random_state": int(random_state),
            "n_to_add_requested": int(n_to_add),
            "minority_count": int(n_min),
            "majority_count": int(n_maj),
            "n_components": int(self.n_components),
        }

        if n_min < 3 or n_to_add <= 0:
            diag["note"] = "no_resampling_needed_or_minority_too_small"
            diag["components"] = []
            return X, y, diag

        # Fit GMM on FULL feature space (all classes)
        K = max(2, int(self.n_components))
        K = min(K, max(2, min(10, X.shape[0] // 30 + 2)))
        gmm = GaussianMixture(
            n_components=K,
            covariance_type="full",
            random_state=int(random_state),
            reg_covar=float(self.reg_covar),
        )
        z = gmm.fit_predict(X)

        comps: List[Dict[str, Any]] = []
        w = np.zeros(K, dtype=float)

        for k in range(K):
            idx = np.where(z == k)[0]
            if idx.size == 0:
                comps.append({"k": int(k), "n": 0})
                continue
            p_k = float(np.mean(y[idx] == 1))
            H_k = float(binary_entropy(np.array([p_k]))[0])
            w_k = (H_k ** float(self.alpha)) * (1.0 - p_k)
            w[k] = w_k
            comps.append({
                "k": int(k),
                "n": int(idx.size),
                "p_k": float(p_k),
                "H_k": float(H_k),
                "w_k": float(w_k),
                "n_min_in_comp": int(np.sum(y[idx] == 1)),
            })

        diag["components"] = comps

        viable = np.array([c.get("n_min_in_comp", 0) >= 3 for c in comps], dtype=bool)
        w = np.where(viable, w, 0.0)

        if w.sum() <= 0:
            return self._global_minority_fallback(X, y, n_to_add, rng, diag)

        alloc = w / w.sum()
        G = int(n_to_add)

        G_k = np.floor(G * alloc).astype(int)
        rem = G - int(G_k.sum())
        if rem > 0:
            order = np.argsort(-alloc)
            for i in range(rem):
                G_k[order[i % len(order)]] += 1

        X_new_list: List[np.ndarray] = []

        for k in range(K):
            if G_k[k] <= 0:
                continue
            idx_comp = np.where(z == k)[0]
            if idx_comp.size == 0:
                continue

            idx_min_comp = idx_comp[y[idx_comp] == 1]
            if idx_min_comp.size < 3:
                continue

            X_comp = X[idx_comp]
            y_comp = y[idx_comp]

            # kNN within component for class-mixture around minority anchors
            k_local = min(self.k_neighbors, max(1, idx_comp.size - 1))
            nn_comp = NearestNeighbors(n_neighbors=k_local + 1).fit(X_comp)

            # minority-to-minority neighbors for interpolation
            X_min_comp = X[idx_min_comp]
            k_eff = min(self.k_neighbors, max(1, X_min_comp.shape[0] - 1))
            nn_min = NearestNeighbors(n_neighbors=k_eff + 1).fit(X_min_comp)
            neigh_min = nn_min.kneighbors(X_min_comp, return_distance=False)[:, 1:]

            neigh_all = nn_comp.kneighbors(X_min_comp, return_distance=False)[:, 1:]
            p_i = np.array([np.mean(y_comp[nn_ids] == 1) for nn_ids in neigh_all], dtype=float)
            H_i = binary_entropy(p_i)

            probs = (H_i / H_i.sum()) if float(H_i.sum()) > 0 else np.ones_like(H_i) / len(H_i)
            chosen = rng.choice(len(idx_min_comp), size=int(G_k[k]), replace=True, p=probs)

            for ii in chosen:
                x_i = X_min_comp[ii]
                nn_idx = int(rng.choice(neigh_min[ii]))
                x_nn = X_min_comp[nn_idx]
                lam = float(rng.random())
                X_new_list.append(x_i + lam * (x_nn - x_i))

        if not X_new_list:
            return self._global_minority_fallback(X, y, n_to_add, rng, diag)

        X_new = np.vstack(X_new_list)
        y_new = np.ones(X_new.shape[0], dtype=int)
        diag["n_generated"] = int(X_new.shape[0])
        diag["note"] = "ok"
        return np.vstack([X, X_new]), np.concatenate([y, y_new]), diag

    def _global_minority_fallback(self, X: np.ndarray, y: np.ndarray, G: int, rng: np.random.Generator, diag: Dict[str, Any]):
        idx_min = np.where(y == 1)[0]
        if idx_min.size < 3 or G <= 0:
            diag["note"] = "fallback_failed_minority<3_or_G<=0"
            diag["n_generated"] = 0
            return X, y, diag

        X_min = X[idx_min]
        k_eff = min(self.k_neighbors, max(1, idx_min.size - 1))
        nn_min = NearestNeighbors(n_neighbors=k_eff + 1).fit(X_min)
        neigh_min = nn_min.kneighbors(X_min, return_distance=False)[:, 1:]

        chosen = rng.choice(len(idx_min), size=int(G), replace=True)
        X_new = []
        for ii in chosen:
            x_i = X_min[ii]
            nn_idx = int(rng.choice(neigh_min[ii]))
            x_nn = X_min[nn_idx]
            lam = float(rng.random())
            X_new.append(x_i + lam * (x_nn - x_i))

        X_new = np.vstack(X_new) if X_new else np.empty((0, X.shape[1]), dtype=float)
        y_new = np.ones(X_new.shape[0], dtype=int)
        diag["note"] = "global_minority_fallback_used"
        diag["n_generated"] = int(X_new.shape[0])
        return np.vstack([X, X_new]), np.concatenate([y, y_new]), diag


@dataclass
class LEO_MinorityPosterior:
    n_components: int = 4
    k_neighbors: int = 5
    alpha: float = 1.0
    c_beta: float = 2.0
    reg_covar: float = 1e-6
    normalize_component_entropy: bool = True

    def fit_resample(self, X: np.ndarray, y: np.ndarray, n_to_add: int, random_state: int = 11) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        rng = _rng(random_state)

        n_min = int(np.sum(y == 1))
        n_maj = int(np.sum(y == 0))
        diag: Dict[str, Any] = {
            "mode": "minority_posterior",
            "random_state": int(random_state),
            "n_to_add_requested": int(n_to_add),
            "minority_count": int(n_min),
            "majority_count": int(n_maj),
            "n_components": int(self.n_components),
        }

        if n_min < 3 or n_to_add <= 0:
            diag["note"] = "no_resampling_needed_or_minority_too_small"
            diag["components"] = []
            return X, y, diag

        idx_min_all = np.where(y == 1)[0]
        X_min_all = X[idx_min_all]

        K = max(2, int(self.n_components))
        K = min(K, max(2, min(10, X_min_all.shape[0] // 10)))
        gmm = GaussianMixture(
            n_components=K,
            covariance_type="full",
            random_state=int(random_state),
            reg_covar=float(self.reg_covar),
        )
        gmm.fit(X_min_all)

        gamma = gmm.predict_proba(X_min_all)
        gamma = np.clip(gamma, EPS, 1.0)
        gamma = gamma / gamma.sum(axis=1, keepdims=True)

        pi = gamma.mean(axis=0)
        Hk = -np.sum(gamma * safe_log(gamma), axis=0)
        if self.normalize_component_entropy:
            Hk = Hk / max(n_min, 1)

        z_hard = np.argmax(gamma, axis=1)
        comp_counts = np.array([(z_hard == k).sum() for k in range(K)], dtype=int)
        viable = comp_counts >= 3

        w = (Hk ** float(self.alpha)) * (1.0 - pi)
        w = np.where(viable, w, 0.0)
        if w.sum() <= 0 and viable.any():
            w = viable.astype(float)

        if w.sum() <= 0:
            diag["note"] = "no_viable_components"
            diag["n_generated"] = 0
            return X, y, diag

        alloc = w / w.sum()
        G = int(n_to_add)

        G_k = np.floor(G * alloc).astype(int)
        rem = G - int(G_k.sum())
        if rem > 0:
            order = np.argsort(-alloc)
            for i in range(rem):
                G_k[order[i % len(order)]] += 1

        comps: List[Dict[str, Any]] = []
        for k in range(K):
            comps.append({
                "k": int(k),
                "n_min_in_comp": int(comp_counts[k]),
                "pi_k": float(pi[k]),
                "Hk": float(Hk[k]),
                "w_k": float(w[k]),
                "G_k": int(G_k[k]),
                "viable": bool(viable[k]),
            })
        diag["components"] = comps

        X_new_list: List[np.ndarray] = []

        for k in range(K):
            if G_k[k] <= 0 or not viable[k]:
                continue

            idx_local = np.where(z_hard == k)[0]
            X_min_k = X_min_all[idx_local]
            if X_min_k.shape[0] < 3:
                continue

            k_eff = min(self.k_neighbors, max(1, X_min_k.shape[0] - 1))
            nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(X_min_k)
            dist, neigh = nn.kneighbors(X_min_k, return_distance=True)
            dist = dist[:, 1:]
            neigh = neigh[:, 1:]

            sigma = np.median(dist[dist > 0]) if np.any(dist > 0) else 1.0
            sigma = float(max(sigma, 1e-6))
            W = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
            W = np.clip(W, EPS, None)
            P = W / W.sum(axis=1, keepdims=True)
            H_i = -np.sum(P * safe_log(P), axis=1) / max(np.log(P.shape[1]), 1.0)

            probs = (H_i / H_i.sum()) if float(H_i.sum()) > 0 else np.ones_like(H_i) / len(H_i)
            chosen = rng.choice(X_min_k.shape[0], size=int(G_k[k]), replace=True, p=probs)

            for ii in chosen:
                x_i = X_min_k[ii]
                nn_idx = int(rng.choice(neigh[ii]))
                x_nn = X_min_k[nn_idx]
                h = float(H_i[ii])
                a = b = 1.0 + float(self.c_beta) * (1.0 - h)
                lam = float(rng.beta(a, b))
                X_new_list.append(x_i + lam * (x_nn - x_i))

        if not X_new_list:
            diag["note"] = "generation_failed"
            diag["n_generated"] = 0
            return X, y, diag

        X_new = np.vstack(X_new_list)
        y_new = np.ones(X_new.shape[0], dtype=int)
        diag["note"] = "ok"
        diag["n_generated"] = int(X_new.shape[0])
        return np.vstack([X, X_new]), np.concatenate([y, y_new]), diag


def main():
    parser = argparse.ArgumentParser(description="LEO oversampling tool: CSV in → oversampled CSV out")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", default=None, help="Path to output CSV (default: <input>_LEO.csv)")
    parser.add_argument("--seed", type=int, default=11, help="Random seed")
    parser.add_argument("--mode", choices=["submitted", "minority_posterior"], default="submitted",
                        help="Which LEO definition to run (default: submitted)")
    parser.add_argument("--target_pct", type=float, default=None,
                        help="Target minority as %% of majority (e.g., 100=balance). If omitted, you'll be prompted.")
    args = parser.parse_args()

    in_path = args.input.strip().strip('"').strip("'")
    if not os.path.exists(in_path):
        print("File not found:", in_path)
        sys.exit(1)

    df = pd.read_csv(in_path)
    print("\nLoaded:", in_path)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    class_col = prompt_choice("\nSelect the class (label) column:", list(df.columns))
    feature_cols = [c for c in df.columns if c != class_col]

    if prompt_yes_no("\nDo you want to drop ID-like columns (recommended)?", default_yes=True):
        suggestions = [c for c in feature_cols if re.search(r"(id$|_id$|^id$|passengerid$)", c.lower())]
        if suggestions:
            print("\nSuggested ID columns:", suggestions)

        drop_cols: List[str] = []
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

    if args.target_pct is None:
        while True:
            s = input("\nTarget minority as % of majority (100=balance, 50=half, 200=twice): ").strip()
            try:
                target_pct = float(s)
                if target_pct >= 0:
                    break
            except ValueError:
                pass
            print("Invalid number. Try again.")
    else:
        target_pct = float(args.target_pct)

    if args.output:
        out_path = args.output
    else:
        base, _ = os.path.splitext(in_path)
        out_path = base + "_LEO.csv"

    y_str = df[class_col].astype(str)
    y = np.where(
        y_str == minority_label, 1,
        np.where(y_str == majority_label, 0, np.nan)
    )
    if np.isnan(y).any():
        print("\nBinary-only tool: class column contains labels beyond the two specified.")
        sys.exit(1)

    y = y.astype(int)
    n_min = int((y == 1).sum())
    n_maj = int((y == 0).sum())
    print(f"\nBEFORE: minority={n_min}, majority={n_maj}, IR={n_maj/max(n_min,1):.3f}")

    target_min = int(round((target_pct / 100.0) * n_maj))
    n_to_add = max(0, target_min - n_min)

    print(f"Target minority count: {target_min} ({target_pct:.1f}% of majority)")
    print(f"Will add: {n_to_add} synthetic minority samples")
    print(f"Mode: {args.mode}")

    X_df = df.drop(columns=[class_col])
    pre = build_preprocessor(X_df)
    X_work = pre.fit_transform(X_df)

    if args.mode == "submitted":
        leo = LEO_Submitted(n_components=4, k_neighbors=5, alpha=1.0, reg_covar=1e-6)
    else:
        leo = LEO_MinorityPosterior(n_components=4, k_neighbors=5, alpha=1.0, c_beta=2.0, reg_covar=1e-6)

    X_res, y_res, diag = leo.fit_resample(X_work, y, n_to_add=n_to_add, random_state=args.seed)

    X_back = decode_back(pre, X_res)
    y_back = np.where(y_res == 1, minority_label, majority_label)

    out_df = X_back.copy()
    out_df[class_col] = y_back
    out_df.to_csv(out_path, index=False)

    n_min2 = int((y_res == 1).sum())
    n_maj2 = int((y_res == 0).sum())
    print(f"\nAFTER : minority={n_min2}, majority={n_maj2}, IR={n_maj2/max(n_min2,1):.3f}")
    print("Saved:", out_path)

    try:
        import json
        diag_path = os.path.splitext(out_path)[0] + "_diag.json"
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2)
        print("Diagnostics:", diag_path)
    except Exception as e:
        print("Could not write diagnostics JSON:", str(e))


if __name__ == "__main__":
    main()
