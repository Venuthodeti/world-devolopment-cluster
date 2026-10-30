#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ============================================================
#  World Development Clusters - Streamlit app
#  Run with:  streamlit run app.py
#  Keep these files in the same folder as app.py:
#  scaler.pkl, pca.pkl, kmeans.pkl, gmm.pkl, app_info.pkl, country_results.csv (or .xls)
# ============================================================
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="World Development Clusters", page_icon="🌍", layout="wide")

# ---------- Settings you can edit ----------
NAMES = {   # friendly cluster names (cluster numbers differ between the two models)
    "K-Means": {0: "Small mid-income states", 1: "Developed countries",
                2: "Large emerging economies", 3: "Least developed countries"},
    "Gaussian Mixture": {0: "Least developed countries", 1: "Developed countries",
                         2: "Small island states", 3: "Middle-income countries"},
}
COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]        # one colour per cluster number
MODELS = {"K-Means": ("kmeans", "Cluster_KMeans"), "Gaussian Mixture": ("gmm", "Cluster_GMM")}
KEY = ["Life Expectancy Female", "Infant Mortality Rate", "Birth Rate",
       "Internet Usage", "Health Exp/Capita", "Population 65+"]   # indicators shown as sliders

# ---------- Load the saved files ----------
here = Path(__file__).resolve().parent
folder = next((p for p in [here, Path.cwd()] if (p / "scaler.pkl").exists()), None)
table = next((folder / n for n in ["country_results.csv", "country_results.xls"]
              if folder and (folder / n).exists()), None)
if folder is None or table is None:
    st.error("Files not found. Put the .pkl files and country_results.csv in the same folder as app.py.")
    st.stop()

@st.cache_resource
def load_models():
    return {n: joblib.load(folder / f"{n}.pkl") for n in ["scaler", "pca", "kmeans", "gmm", "app_info"]}

@st.cache_data
def load_table():
    return pd.read_csv(table, index_col=0)

m, countries = load_models(), load_table()
scaler, pca = m["scaler"], m["pca"]
features, log_cols = m["app_info"]["columns"], m["app_info"]["log_cols"]

# ---------- Sidebar: choose model and starting country ----------
with st.sidebar:
    st.title("⚙️ Settings")
    model_name = st.radio("Clustering model", list(MODELS))
    start = st.selectbox("Start from a country", ["Typical country"] + sorted(countries.index))
    st.caption("Pick a real country, then change its indicators to see how its cluster changes.")

model, col = m[MODELS[model_name][0]], MODELS[model_name][1]
name = lambda c: NAMES[model_name].get(int(c), f"Cluster {int(c)}")
color_map = {name(c): COLORS[c % 4] for c in countries[col].unique()}

# ---------- Header ----------
st.markdown("""
<style>
.hero {background: linear-gradient(90deg,#1f77b4,#2ca02c); padding: 1.5rem 2rem; border-radius: 14px; color: white;}
.hero h1 {margin: 0; color: white;}
.result {padding: 1rem 1.5rem; border-radius: 14px; color: white; margin-bottom: 1rem;}
.result h2 {margin: 0; color: white;}
</style>
<div class="hero"><h1>🌍 World Development Clusters</h1>
<p>Which group of countries does a country belong to? Explore it with machine learning.</p></div>
""", unsafe_allow_html=True)
st.write("")
a, b, c, d = st.columns(4)
a.metric("Countries", len(countries)); b.metric("Clusters", countries[col].nunique())
c.metric("Indicators", len(features)); d.metric("Model", model_name)

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "🗺️ World Map", "📊 Cluster Profiles", "ℹ️ About"])

# ======================= TAB 1: PREDICT =======================
with tab1:
    median = countries[features].median()
    base = median if start == "Typical country" else countries.loc[start, features].fillna(median)

    left, right = st.columns(2)
    with left:
        st.subheader("1. Set the indicators")
        values = {}
        for f in KEY:                                   # main indicators as sliders
            lo, hi = float(countries[f].min()), float(countries[f].max())
            values[f] = st.slider(f, lo, hi, float(base[f]), step=(hi - lo) / 100,
                                  format="%.3g", key=f"{f}|{start}")
        with st.expander("More indicators"):            # the other indicators as number boxes
            for f in [f for f in features if f not in KEY]:
                values[f] = st.number_input(f, value=float(f"{base[f]:.6g}"), format="%g",
                                            min_value=0.0 if f in log_cols else None, key=f"{f}|{start}")

    # Prepare the input like the notebook: log -> scale -> PCA, then ask the model
    X = pd.DataFrame([values])[features]
    X[log_cols] = np.log1p(X[log_cols])
    point = pca.transform(scaler.transform(X))
    label = int(model.predict(point)[0])

    with right:
        st.subheader("2. Result")
        st.markdown(f'<div class="result" style="background:{COLORS[label % 4]}">'
                    f'This country belongs to<h2>{name(label)}</h2></div>', unsafe_allow_html=True)

        pcs = countries[[c for c in countries.columns if c.startswith("PC")]].to_numpy()
        nearest = countries.index[np.argsort(np.linalg.norm(pcs - point, axis=1))[:5]]
        st.write("**🤝 Most similar countries:** " + ", ".join(nearest))

        plot = pd.DataFrame({"PC1": pcs[:, 0], "PC2": pcs[:, 1], "Country": countries.index,
                             "Cluster": countries[col].map(name)})
        fig = px.scatter(plot, x="PC1", y="PC2", color="Cluster", hover_name="Country",
                         color_discrete_map=color_map)
        fig.add_scatter(x=[point[0, 0]], y=[point[0, 1]], mode="markers", name="Your input",
                        marker=dict(size=20, symbol="star", color="black"))
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0), legend_title_text="")
        st.plotly_chart(fig, width="stretch")

# ======================= TAB 2: WORLD MAP =======================
with tab2:
    map_df = countries.reset_index(names="Country")
    map_df["Cluster"] = map_df[col].map(name)
    fig = px.choropleth(map_df, locations="Country", locationmode="country names", color="Cluster",
                        hover_name="Country", color_discrete_map=color_map)
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0), legend_title_text="")
    st.plotly_chart(fig, width="stretch")
    pick = st.selectbox("Browse a cluster", sorted(color_map))
    st.write(", ".join(map_df.loc[map_df["Cluster"] == pick, "Country"]))

# ======================= TAB 3: CLUSTER PROFILES =======================
with tab3:
    sizes = countries[col].map(name).value_counts().rename_axis("Cluster").reset_index(name="Countries")
    profile = countries.groupby(col)[features].median()
    zscore = (profile - profile.mean()) / profile.std()        # high / low compared with other clusters
    zscore.index = [name(i) for i in zscore.index]

    c1, c2 = st.columns([1, 2])
    c1.subheader("Countries per cluster")
    c1.plotly_chart(px.bar(sizes, x="Cluster", y="Countries", color="Cluster", color_discrete_map=color_map)
                    .update_layout(showlegend=False, height=520), width="stretch")
    c2.subheader("What each cluster looks like")
    c2.plotly_chart(px.imshow(zscore.T, color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                              aspect="auto").update_layout(height=520), width="stretch")
    st.caption("Red = high, blue = low (compared with the other clusters).")

# ======================= TAB 4: ABOUT =======================
with tab4:
    st.markdown("""
    ### How it works
    1. **Data:** World Development indicators (economy, health, population, technology) for 208 countries.
    2. **Cleaning:** text values converted to numbers, one row per country, skewed indicators log-transformed.
    3. **Preparation:** missing values filled, features scaled, PCA keeps 95% of the information.
    4. **Models compared:** K-Means, DBSCAN, Gaussian Mixture and Hierarchical (Silhouette, Davies-Bouldin, Calinski-Harabasz).
    5. **This app:** repeats the same steps for any country and asks K-Means or Gaussian Mixture for its cluster.
    """)

