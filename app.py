import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')

import viz_extensions as vext

st.set_page_config(page_title="Palmer Penguins Insight Hub", layout="wide", page_icon="🐧")

# Custom CSS for Premium Design
st.markdown("""
    <style>
    .main {
        background-color: #0d1117;
        color: #e6edf3;
    }
    h1, h2, h3 {
        color: #58a6ff;
    }
    .stMetric {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 10px;
    }
    div[data-testid="stMetricLabel"],
    label[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] *,
    label[data-testid="stMetricLabel"] *,
    div[data-testid="stMetricLabel"] p,
    label[data-testid="stMetricLabel"] p {
        color: #ffffff !important;
        opacity: 1 !important;
    }
    div[data-testid="stMetricValue"],
    label[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] *,
    label[data-testid="stMetricValue"] *,
    div[data-testid="stMetricValue"] p,
    label[data-testid="stMetricValue"] p {
        color: #ffffff !important;
        opacity: 1 !important;
    }
    div[data-testid="stMetricDelta"],
    label[data-testid="stMetricDelta"],
    div[data-testid="stMetricDelta"] *,
    label[data-testid="stMetricDelta"] * {
        color: #ffffff !important;
        opacity: 1 !important;
    }
    .insight-box {
        background: linear-gradient(135deg, #1f6feb 0%, #3fb950 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        margin-bottom: 20px;
        border-left: 5px solid #fff;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prep_data():
    df = sns.load_dataset('penguins')
    # Handle Missing Values
    df_clean = df.dropna(subset=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex'])
    
    # Feature Engineering -> Calculate the Bill Ratio 
    df_clean['bill_ratio'] = df_clean['bill_length_mm'] / df_clean['bill_depth_mm']
    
    return df_clean

@st.cache_data
def apply_ml_pipeline(df):
    if df.empty:
        return df, None, None
        
    # Feature Selection 
    features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    X = df[features]
    
    # StandardScaler 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means Clustering - Fallback to min clusters based on data
    n_clusters = min(3, len(df))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['ml_cluster'] = kmeans.fit_predict(X_scaled)
    df['ml_cluster'] = df['ml_cluster'].astype(str) 
    
    # PCA 
    n_components = min(2, len(df), len(features))
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(X_scaled)
    df['pca_1'] = pca_results[:, 0]
    if n_components > 1:
        df['pca_2'] = pca_results[:, 1]
    else:
        df['pca_2'] = 0
    
    # PCA load component ratios
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    return df, pca, loadings


def _apply_white_labels(fig):
    fig.update_layout(
        title_font=dict(color='white'),
        legend=dict(font=dict(color='white'), title_font=dict(color='white')),
        xaxis=dict(
            color='white',
            tickfont=dict(color='white'),
            title_font=dict(color='white'),
        ),
        yaxis=dict(
            color='white',
            tickfont=dict(color='white'),
            title_font=dict(color='white'),
        ),
    )
    return fig

def plot_interactive_ml_scatter(df):
    # Tạo biểu đồ Scatter với trục x, y (Đât sẽ là 2 thành phần PCA)
    fig = px.scatter(
        df, 
        x="pca_1", 
        y="pca_2", 
        color="ml_cluster", 
        symbol="species", 
        size="body_mass_g", 
        hover_name="species",
        hover_data={
            "pca_1": False, 
            "pca_2": False,
            "island": True,
            "sex": True,
            "body_mass_g": True,
            "bill_ratio": ":.2f", 
            "ml_cluster": True
        },
        title="PCA & K-Means Clustering of Palmer Penguins (Hover for Profile)",
        labels={"ml_cluster": "K-Means Cluster"},
        color_discrete_sequence=px.colors.qualitative.Vivid
    )

    # Sliders/Buttons trực tiếp trong Plotly
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.1,
                y=1.15,
                buttons=list([
                    dict(label="All",
                         method="update",
                         args=[{"visible": [True, True, True]}]),
                    dict(label="Male Only",
                         method="update",
                         args=[{"visible": [True if s == 'Male' else False for s in df['sex'].unique()]}]),
                    dict(label="Female Only",
                         method="update",
                         args=[{"visible": [True if s == 'Female' else False for s in df['sex'].unique()]}])
                ]),
                bgcolor="white",
                font=dict(color="black")
            )
        ],
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        font=dict(color='white')
    )
    _apply_white_labels(fig)

    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    return fig

def main():
    st.title("🐧 The Palmer Penguins Insight Hub")
    st.markdown("Exploring biological relationships and physical attributes of penguin species across the Palmer Archipelago.")
    
    # Load data
    df_base = load_and_prep_data()
    
    # Sidebar Filtering
    st.sidebar.header("Filter Data")
    species_options = df_base['species'].unique().tolist()
    island_options = df_base['island'].unique().tolist()
    sex_options = df_base['sex'].unique().tolist()
    
    selected_species = st.sidebar.multiselect("Select Species", species_options, default=species_options)
    selected_islands = st.sidebar.multiselect("Select Islands", island_options, default=island_options)
    selected_sex = st.sidebar.multiselect("Select Sex", sex_options, default=sex_options)
    
    # Apply filters
    filtered_df = df_base[
        (df_base['species'].isin(selected_species)) &
        (df_base['island'].isin(selected_islands)) &
        (df_base['sex'].isin(selected_sex))
    ].copy()

    _bm = filtered_df["body_mass_g"]
    _lo, _hi = int(_bm.min()), int(_bm.max())
    if _lo < _hi:
        _mass_range = st.sidebar.slider("Body mass range (g)", _lo, _hi, (_lo, _hi))
    else:
        _mass_range = (_lo, _hi)
    filtered_df = filtered_df[
        (filtered_df["body_mass_g"] >= _mass_range[0])
        & (filtered_df["body_mass_g"] <= _mass_range[1])
    ].copy()

    if filtered_df.empty:
        st.warning("No data found for the selected filters.")
        return
        
    df_ml, pca_model, pca_loadings = apply_ml_pipeline(filtered_df)
    
    # Top Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Penguins", len(filtered_df))
    col2.metric("Avg Body Mass", f"{filtered_df['body_mass_g'].mean():.1f} g")
    col3.metric("Avg Flipper Length", f"{filtered_df['flipper_length_mm'].mean():.1f} mm")
    col4.metric("Avg Bill Ratio", f"{filtered_df['bill_ratio'].mean():.2f}")
    
    # Dynamic Insight Box (Storytelling context)
    avg_mass_text = f"{filtered_df['body_mass_g'].mean():.0f}g"
    species_text = " and ".join(selected_species) if len(selected_species) <= 2 else "Multiple Species"
    island_text = " and ".join(selected_islands) if len(selected_islands) <= 2 else "Multiple Islands"
    
    st.markdown(f'<div class="insight-box">💡 Insight: On {island_text}, the selected {species_text} penguins average {avg_mass_text} in mass.<br>Did you know that bill depth and flipper length play crucial roles in their cluster separation? Toggle the clusters below!</div>', unsafe_allow_html=True)


    st.markdown("---")
    st.header("0. Core exploratory visualizations & narrative")
    st.markdown(vext.build_story_markdown(filtered_df))
    _row_a = st.columns(3)
    with _row_a[0]:
        st.plotly_chart(vext.create_scatter_plot(filtered_df), use_container_width=True)
    with _row_a[1]:
        st.plotly_chart(vext.create_bar_chart(filtered_df, "species"), use_container_width=True)
    with _row_a[2]:
        st.plotly_chart(vext.create_bar_chart(filtered_df, "island"), use_container_width=True)
    _row_b = st.columns(2)
    with _row_b[0]:
        st.plotly_chart(
            vext.create_distribution_plot(filtered_df, "flipper_length_mm"),
            use_container_width=True,
        )
    with _row_b[1]:
        st.plotly_chart(vext.create_pca_plot(df_ml), use_container_width=True)
    st.caption(
        "PCA plot uses PC1/PC2 from the same standardized features as the ML pipeline above. "
        "`viz_extensions.run_kmeans()` mirrors KMeans+StandardScaler for reuse in notebooks or reports."
    )
    st.plotly_chart(vext.create_kmeans_bill_scatter(df_ml), use_container_width=True)

    # Core Visuals & ML Vis
    st.header("1. K-Means Clusters in 3D Original Feature Space")
    
    fig_3d = px.scatter_3d(
        df_ml,
        x='bill_length_mm',
        y='flipper_length_mm',
        z='body_mass_g',
        color='ml_cluster', 
        symbol='species', 
        title="3D Feature Plot (Color=Cluster, Shape=Real Species)",
        hover_data=['sex', 'island', 'bill_ratio'],
        labels={
            'bill_length_mm': 'Bill Length (mm)',
            'flipper_length_mm': 'Flipper Length (mm)',
            'body_mass_g': 'Body Mass (g)',
            'ml_cluster': 'K-Means Cluster'
        },
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig_3d.update_traces(marker=dict(size=5, line=dict(width=1, color='DarkSlateGrey')))
    fig_3d.update_layout(
        scene=dict(
            bgcolor='#0d1117',
            xaxis=dict(color='white', title_font=dict(color='white'), tickfont=dict(color='white')),
            yaxis=dict(color='white', title_font=dict(color='white'), tickfont=dict(color='white')),
            zaxis=dict(color='white', title_font=dict(color='white'), tickfont=dict(color='white')),
        ),
        title_font=dict(color='white'),
        legend=dict(font=dict(color='white'), title_font=dict(color='white')),
        paper_bgcolor='#0d1117',
        font=dict(color='white'),
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    
    st.header("1.1 Dashboard dynamics")
    fig_interactive = plot_interactive_ml_scatter(df_ml)
    st.plotly_chart(fig_interactive, use_container_width=True)
    
    st.header("2. PCA Feature Loadings")
    features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    
    if pca_loadings is not None:
        fig_pca_loadings = go.Figure()
        # Adding feature vectors
        for i, feature in enumerate(features):
            fig_pca_loadings.add_annotation(
                x=pca_loadings[i, 0], y=pca_loadings[i, 1] if pca_loadings.shape[1] > 1 else 0,
                ax=0, ay=0,
                xanchor="center", yanchor="bottom",
                text=feature,
                font=dict(size=14, color="#ff4b4b")
            )
            fig_pca_loadings.add_shape(
                type='line',
                x0=0, y0=0, x1=pca_loadings[i, 0], y1=pca_loadings[i, 1] if pca_loadings.shape[1] > 1 else 0,
                line=dict(color="#ff4b4b", width=2, dash="dot")
            )
            
        fig_pca_loadings.update_layout(
            title="Which physical traits drive the separation?",
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            xaxis=dict(
                range=[-1.2, 1.2],
                zerolinecolor='gray',
                showgrid=False,
                color='white',
                tickfont=dict(color='white'),
                title_font=dict(color='white'),
            ),
            yaxis=dict(
                range=[-1.2, 1.2],
                zerolinecolor='gray',
                showgrid=False,
                color='white',
                tickfont=dict(color='white'),
                title_font=dict(color='white'),
            ),
            title_font=dict(color='white'),
            width=700, height=500,
            paper_bgcolor='#0d1117',
            plot_bgcolor='#0d1117',
            font=dict(color='white')
        )
        st.plotly_chart(fig_pca_loadings, use_container_width=True)
    
    # Basic Data Exploration
    st.header("3. Distributions & Correlations")
    col1, col2 = st.columns(2)
    with col1:
        fig_hist = px.histogram(df_ml, x='body_mass_g', color='species', barmode='overlay', title='Body Mass Distribution by Species',
                               color_discrete_map=vext.SPECIES_COLORS)
        fig_hist.update_layout(paper_bgcolor='#0d1117', plot_bgcolor='#0d1117', font=dict(color='white'))
        _apply_white_labels(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)
    with col2:
        fig_scatter = px.scatter(df_ml, x='flipper_length_mm', y='body_mass_g', color='species', size='bill_ratio', hover_data=['island'],
                                 title='Flipper Length vs Body Mass', color_discrete_map=vext.SPECIES_COLORS)
        fig_scatter.update_layout(paper_bgcolor='#0d1117', plot_bgcolor='#0d1117', font=dict(color='white'))
        _apply_white_labels(fig_scatter)
        st.plotly_chart(fig_scatter, use_container_width=True)

if __name__ == "__main__":
    main()
