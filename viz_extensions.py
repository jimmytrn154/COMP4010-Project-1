from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def create_scatter_plot(df: pd.DataFrame) -> go.Figure:
    """
    2D scatter: bill_length_mm vs bill_depth_mm, colored by species (interactive).
    Adds a short annotation summarizing separation in bill space for the current slice.
    """
    fig = px.scatter(
        df,
        x="bill_length_mm",
        y="bill_depth_mm",
        color="species",
        hover_data=["island", "sex", "body_mass_g", "flipper_length_mm"],
        title="Bill length vs bill depth (colored by species)",
        labels={
            "bill_length_mm": "Bill length (mm)",
            "bill_depth_mm": "Bill depth (mm)",
            "species": "Species",
        },
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=0.5, color="white")))
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="white"),
        legend_title_text="Species",
    )

    if not df.empty and "species" in df.columns:
        means = df.groupby("species", observed=True)[["bill_length_mm", "bill_depth_mm"]].mean()
        if len(means) >= 2:
            s1, s2 = means.index[0], means.index[1]
            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="left",
                text=(
                    f"Mean bill length: {s1} {means.loc[s1, 'bill_length_mm']:.1f} mm vs "
                    f"{s2} {means.loc[s2, 'bill_length_mm']:.1f} mm (filtered data)."
                ),
                font=dict(size=11, color="#8b949e"),
            )
    return fig


def create_bar_chart(df: pd.DataFrame, by: str = "island") -> go.Figure:
    """
    Bar chart of penguin counts. `by` is either 'island' or 'species'.
    """
    if by not in ("island", "species"):
        by = "island"
    if df.empty or by not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            title="Bar chart (no data)",
            paper_bgcolor="#0d1117",
            font=dict(color="white"),
        )
        return fig
    counts = df[by].value_counts().reset_index()
    counts.columns = [by, "count"]
    fig = px.bar(
        counts,
        x=by,
        y="count",
        color=by,
        title=f"Penguin counts by {by.replace('_', ' ')}",
        labels={"count": "Count"},
    )
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="white"),
        showlegend=False,
    )
    fig.update_xaxes(title=by.replace("_", " ").title())
    return fig


def create_distribution_plot(
    df: pd.DataFrame,
    column: str = "flipper_length_mm",
    use_histogram: bool = True,
) -> go.Figure:
    """
    Distribution of a numeric measure (default: flipper_length_mm) with optional
    overlaid histograms by species. Set use_histogram=False to approximate a KDE-like
    view via histnorm='probability density' with more bins (still Plotly-native).
    """
    col = column if column in df.columns else "flipper_length_mm"
    histnorm = None if use_histogram else "probability density"
    fig = px.histogram(
        df,
        x=col,
        color="species",
        barmode="overlay",
        opacity=0.55,
        nbins=35,
        histnorm=histnorm,
        title=f"Distribution of {col.replace('_', ' ')} (by species)",
        labels={col: col.replace("_", " ").replace(" mm", " (mm)").replace(" g", " (g)")},
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="white"),
        bargap=0.05,
    )
    return fig


def run_kmeans(
    df: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
    feature_cols: tuple[str, ...] = (
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ),
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """
    Standardize the four numeric bill/morphology features, then fit KMeans.

    This is the same modeling idea as apply_ml_pipeline() in app.py: scaling puts
    mm-based and gram-based features on comparable scales so clusters reflect shape,
    not raw units. Returns a copy of df with a string column 'kmeans_cluster'.
    """
    use = [c for c in feature_cols if c in df.columns]
    work = df.copy()
    if work.empty or len(use) < 4:
        work["kmeans_cluster"] = pd.Series(dtype="str")
        return work, KMeans(n_clusters=1), StandardScaler()

    X = work[use].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    k = min(n_clusters, len(work))
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    work["kmeans_cluster"] = km.fit_predict(Xs).astype(str)
    return work, km, scaler


def create_pca_plot(df: pd.DataFrame) -> go.Figure:
    """
    2D PCA scatter: PC1 vs PC2, colored by species. Expects columns pca_1 and pca_2
    (as produced by apply_ml_pipeline in app.py after scaling + PCA on the same features).
    """
    need = {"pca_1", "pca_2", "species"}
    if df.empty or not need.issubset(df.columns):
        fig = go.Figure()
        fig.update_layout(
            title="PCA plot (insufficient data after filtering)",
            paper_bgcolor="#0d1117",
            font=dict(color="white"),
        )
        return fig

    fig = px.scatter(
        df,
        x="pca_1",
        y="pca_2",
        color="species",
        hover_data=["island", "body_mass_g"],
        title="PCA: PC1 vs PC2 (color = species)",
        labels={"pca_1": "PC1", "pca_2": "PC2"},
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="white")))
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="white"),
    )
    # Explained variance is not passed in; annotate that PC axes come from standardized features
    fig.add_annotation(
        x=0.02,
        y=0.02,
        xref="paper",
        yref="paper",
        showarrow=False,
        text="PCs computed from standardized bill + flipper + mass features.",
        font=dict(size=11, color="#8b949e"),
    )
    return fig


def create_kmeans_bill_scatter(df: pd.DataFrame) -> go.Figure:
    """2D view of KMeans clusters in bill space (length vs depth)."""
    if df.empty or "ml_cluster" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="KMeans bill view (no clusters)", paper_bgcolor="#0d1117")
        return fig
    fig = px.scatter(
        df,
        x="bill_length_mm",
        y="bill_depth_mm",
        color="ml_cluster",
        symbol="species",
        hover_data=["island", "body_mass_g"],
        title="KMeans clusters in bill length vs depth (shape = true species)",
        labels={"ml_cluster": "Cluster"},
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font=dict(color="white"))
    return fig


def compute_standalone_pca_2d(
    df: pd.DataFrame,
    feature_cols: tuple[str, ...] = (
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ),
) -> pd.DataFrame:
    """
    Optional: recompute PCA(2) for teaching/presentation — same math as pipeline, separate function.
    Adds columns pc1_standalone, pc2_standalone without touching teammate pipeline outputs.
    """
    out = df.copy()
    use = [c for c in feature_cols if c in out.columns]
    if out.empty or len(out) < 3 or len(use) < 4:
        return out
    Xs = StandardScaler().fit_transform(out[use])
    pca = PCA(n_components=2, random_state=42)
    z = pca.fit_transform(Xs)
    out["pc1_standalone"] = z[:, 0]
    out["pc2_standalone"] = z[:, 1]
    return out


def build_story_markdown(df: pd.DataFrame) -> str:
    """
    Dynamic markdown bullets from the *current filtered* dataframe: comparisons and island mix.
    """
    if df.empty:
        return "_No rows selected — adjust filters._"

    lines: list[str] = []
    lines.append("### Narrative snapshot (filtered cohort)")
    lines.append("")

    # Flipper comparison: Adelie vs Gentoo if both present
    if "species" in df.columns and "flipper_length_mm" in df.columns:
        g = df.groupby("species", observed=True)["flipper_length_mm"].mean().sort_values()
        if "Adelie" in g.index and "Gentoo" in g.index:
            lines.append(
                f"- **Flipper length:** On average, Adelie ({g['Adelie']:.1f} mm) tend to have "
                f"shorter flippers than Gentoo ({g['Gentoo']:.1f} mm) in this selection."
            )
        elif len(g) >= 2:
            lo, hi = g.iloc[0], g.iloc[-1]
            lines.append(
                f"- **Flipper spread:** Mean flipper length ranges from {lo:.1f} mm to {hi:.1f} mm across species shown."
            )

    # Island dominance: Biscoe / Gentoo style fact
    if "island" in df.columns and "species" in df.columns:
        for isl in df["island"].dropna().unique():
            sub = df[df["island"] == isl]
            top = sub["species"].value_counts()
            if len(top):
                top_sp, n = top.index[0], int(top.iloc[0])
                share = n / len(sub) if len(sub) else 0
                lines.append(
                    f"- **{isl}:** {top_sp} is the most common species ({share:.0%} of penguins on this island in the filter)."
                )

    # Body mass range in selection
    if "body_mass_g" in df.columns:
        lines.append(
            f"- **Body mass in view:** from {df['body_mass_g'].min():.0f} g to {df['body_mass_g'].max():.0f} g "
            f"(mean {df['body_mass_g'].mean():.0f} g)."
        )

    lines.append("")
    lines.append(
        "_These sentences update when you change species, island, sex, or body-mass range._"
    )
    return "\n".join(lines)
