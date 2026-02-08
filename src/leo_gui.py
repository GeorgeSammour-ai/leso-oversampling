import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import re

# Import official LEO logic from your local leso_csv_tool.py
from leso_csv_tool import LESO, build_preprocessor, decode_back

# --- PAGE CONFIG ---
st.set_page_config(page_title="LEO Analytics Workbench", layout="wide")

# --- HEADER & DESCRIPTION ---
st.title("🦁 LEO: Latent Entropy-guided Synthetic Oversampling")
st.markdown("""
### **What is LEO?**
**LEO** (Latent Entropy-guided Synthetic Oversampling) is a sophisticated framework for balancing tabular datasets. 
Unlike standard methods that sample randomly or linearly, LEO uses **Gaussian Mixture Models (GMM)** to understand the 
underlying "latent states" of your data. It calculates the **Binary Entropy** of these states to identify 
high-uncertainty regions (class boundaries) and focuses synthetic data generation there.



### **How to use this tool:**
1. **Upload:** Provide your imbalanced CSV dataset in the sidebar.
2. **Configure:** Identify your Target Column and specify which labels are Minority vs. Majority.
3. **Strategy:** Exclude non-predictive columns (IDs/Names) and set your target balance ratio.
4. **Execute:** Run the engine to see performance metrics, distribution plots, and download your new dataset.
""")
st.divider()

# --- SIDEBAR: INPUTS & PARAMETERS ---
with st.sidebar:
    st.header("📂 1. Data Input")
    uploaded_file = st.file_uploader("Upload your Dataset (CSV)", type="csv")
    
    st.divider()
    st.header("⚙️ 2. LEO Hyperparameters")
    seed = st.number_input("Random Seed", value=11)
    k_neighbors = st.slider("k Neighbors", 1, 20, 5, help="Local density for synthetic generation")
    n_states = st.slider("Max Latent States (GMM)", 2, 10, 4, help="Complexity of cluster detection")
    alpha = st.slider("Entropy Weight (Alpha)", 0.0, 3.0, 1.0)
    c_beta = st.slider("Beta Curvature (c_beta)", 0.0, 5.0, 2.0)

# --- MAIN LOGIC ---
if not uploaded_file:
    st.warning("### 📥 Action Required: Please upload a CSV file to begin.")
    st.info("The configuration and analytics panels will appear here once your data is loaded.")
else:
    df = pd.read_csv(uploaded_file)
    st.success(f"✔️ Successfully loaded: {uploaded_file.name}")
    
    # --- STEP 1: INITIAL STATS ---
    st.subheader("📊 Step 1: Data Configuration & Initial IR")
    col1, col2, col3 = st.columns(3)
    with col1:
        class_col = st.selectbox("Select Target (Label) Column", df.columns)
    
    available_labels = df[class_col].unique().astype(str)
    
    with col2:
        sel_min = st.selectbox("Identify Minority Label", available_labels, index=0)
    with col3:
        sel_maj = st.selectbox("Identify Majority Label", available_labels, index=min(1, len(available_labels)-1))

    # Calculate Counts and Imbalance Ratio (IR)
    n_min_init = len(df[df[class_col].astype(str) == sel_min])
    n_maj_init = len(df[df[class_col].astype(str) == sel_maj])
    
    if n_min_init > 0:
        ir_init = n_maj_init / n_min_init
        s1, s2, s3 = st.columns(3)
        s1.metric("Minority Count", n_min_init)
        s2.metric("Majority Count", n_maj_init)
        s3.metric("Initial IR", f"{ir_init:.2f} : 1")
    
    st.divider()

    # --- STEP 2: STRATEGY & PROJECTION ---
    st.subheader("🎯 Step 2: Resampling Strategy")
    
    id_suggestions = [c for c in df.columns if re.search(r"(id$|_id$|^id$)", c.lower()) and c != class_col]
    drop_cols = st.multiselect("Exclude Columns (IDs, Text, Names)", [c for c in df.columns if c != class_col], default=id_suggestions)
    
    target_pct = st.slider("Target Balance % (Minority as % of Majority)", 10, 300, 100)

    # Real-time Projection Math
    target_count = int(round((target_pct / 100.0) * n_maj_init))
    n_to_be_added = max(0, target_count - n_min_init)
    final_ir = n_maj_init / target_count if target_count > 0 else 0

    # Projected Bar Chart
    chart_data = pd.DataFrame({
        "Class": [sel_maj, sel_min, sel_min],
        "Count": [n_maj_init, n_min_init, n_to_be_added],
        "Source": ["Original", "Original", "LEO Synthetic"]
    })
    
    fig_proj = px.bar(chart_data, x="Class", y="Count", color="Source", 
                      title="Projected Class Distribution",
                      color_discrete_map={"Original": "#636EFA", "LEO Synthetic": "#00CC96"},
                      barmode="stack")
    st.plotly_chart(fig_proj, use_container_width=True)

    if n_to_be_added == 0:
        st.warning(f"⚠️ **Note:** Your minority class already meets the {target_pct}% threshold. No samples will be added.")
    else:
        st.info(f"💡 **Action:** LEO will add **{n_to_be_added}** samples to achieve a Target IR of **{final_ir:.2f} : 1**.")

    # --- STEP 3: EXECUTION ---
    if st.button("🚀 Step 3: Run LEO & Generate Analytics"):
        try:
            with st.spinner("Processing Latent States and Entropy Weights..."):
                # Feature Preparation
                cols_to_process = [c for c in df.columns if c not in drop_cols and c != class_col]
                X_df = df[cols_to_process].copy()
                y = np.where(df[class_col].astype(str) == sel_min, 1, 0)
                
                # Transform Data
                pre = build_preprocessor(X_df)
                X_work = pre.fit_transform(X_df)
                
                # Oversample
                leo = LESO(n_states=n_states, k_neighbors=k_neighbors, alpha=alpha, c_beta=c_beta)
                X_res, y_res = leo.fit_resample(X_work, y, n_to_add=n_to_be_added, random_state=int(seed))
                X_back = decode_back(pre, X_res)

            # --- ANALYTICS DASHBOARD ---
            st.divider()
            st.header("📈 Step 4: Analytics & Export")
            
            with st.expander("📝 Applied Configuration Summary", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("k Neighbors", k_neighbors)
                c2.metric("Latent States", n_states)
                c3.metric("Final IR", f"{final_ir:.2f}:1")
                c4.metric("Samples Created", n_to_be_added)

            # Performance Metrics
            st.subheader("Model Validation (Random Forest)")
            X_train, X_test, y_train, y_test = train_test_split(X_work, y, test_size=0.3, random_state=seed)
            f1_orig = f1_score(y_test, RandomForestClassifier(random_state=seed).fit(X_train, y_train).predict(X_test))
            f1_leo = f1_score(y_test, RandomForestClassifier(random_state=seed).fit(X_res, y_res).predict(X_test))
            
            m1, m2 = st.columns(2)
            m1.metric("Original F1-Score", f"{f1_orig:.3f}")
            m2.metric("LEO Enhanced F1-Score", f"{f1_leo:.3f}", delta=f"{f1_leo - f1_orig:.3f}")

            

            # Visualizations
            st.subheader("Decision Space Mapping")
            t1, t2, t3 = st.tabs(["PCA (Variance)", "t-SNE (Clustering)", "Feature Correlations"])
            lbl = np.where(y_res == 1, sel_min, sel_maj)
            src = ["Original"]*len(y) + ["Synthetic"]*(len(y_res)-len(y))
            
            with t1:
                pca_c = PCA(n_components=2).fit_transform(X_res)
                st.plotly_chart(px.scatter(x=pca_c[:,0], y=pca_c[:,1], color=lbl, symbol=src, title="PCA Global Feature Variance"), use_container_width=True)
            with t2:
                tsne_c = TSNE(n_components=2, random_state=seed).fit_transform(X_res)
                st.plotly_chart(px.scatter(x=tsne_c[:,0], y=tsne_c[:,1], color=lbl, symbol=src, title="t-SNE Local Density Mapping"), use_container_width=True)
            with t3:
                st.plotly_chart(px.imshow(X_back.select_dtypes(include=[np.number]).corr(), color_continuous_scale='RdBu_r'), use_container_width=True)

            # Final Data Reconstruction
            final_df = X_back.copy()
            final_df[class_col] = lbl
            for c in drop_cols:
                final_df[c] = np.concatenate([df[c].values, [np.nan]*(len(final_df)-len(df))])
            
            st.download_button("📥 Download Balanced CSV", final_df.to_csv(index=False), f"leo_resampled_ir_{final_ir:.1f}.csv")

        except Exception as e:
            st.error(f"Processing Error: {e}")