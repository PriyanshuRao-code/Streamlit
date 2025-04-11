import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from io import BytesIO
import os

# =============================================
# HARD REQUIREMENT: Load CSV before anything
FILE_PATH = "my_final.csv"

if not os.path.exists(FILE_PATH):
    st.error(f"CSV file not found at path: {FILE_PATH}")
    st.stop()

# Read CSV
df_results = pd.read_csv(FILE_PATH)

df_results = df_results.fillna("None")

# Check required columns
required_columns = [
    'model',
    'data_preprocessing_methods_outlier',
    'data_preprocessing_methods_encode',
    'data_preprocessing_methods_scale',
    'data_preprocessing_methods_correlation',
    'data_preprocessing_methods_pca',
    'data_preprocessing_methods_select_corr',
    'valid_accuracy', 'valid_precision', 'valid_recall', 'valid_f1_score',
    'test_accuracy', 'test_precision', 'test_recall', 'test_f1_scorea'
]

missing_cols = [col for col in required_columns if col not in df_results.columns]
if missing_cols:
    st.error(f"The following required columns are missing in your CSV:\n\n{missing_cols}")
    st.stop()

# =============================================
# Helper functions to download images
def download_plotly_fig(fig, filename):
    img_bytes = fig.to_image(format="png")
    st.download_button(
        label="📥 Download Image",
        data=img_bytes,
        file_name=filename,
        mime="image/png"
    )

def download_matplotlib_fig(fig, filename):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="📥 Download Image",
        data=buf,
        file_name=filename,
        mime="image/png"
    )

# =============================================
# UI
st.set_page_config(page_title="ML Dashboard", layout="wide")
st.title("🚀 Asteroid ML Project Dashboard")

st.markdown("### Preview of Loaded Data")
st.dataframe(df_results.head())

# =============================================
# Mode Selection
mode = st.sidebar.selectbox("📊 Select Dashboard Mode", [
    "Dashboard 1: Compare All Models (on a preprocessing setup)",
    "Dashboard 2: Compare All Pipelines (for one model)"
])

# =============================================
# DASHBOARD 1
if "Dashboard 1" in mode:
    st.header("📋 Dashboard 1: Model Comparison for Selected Preprocessing")

    # Sidebar filters
    outlier_option = st.sidebar.selectbox("Outlier", df_results["data_preprocessing_methods_outlier"].unique())
    encode_option = st.sidebar.selectbox("Encoding", df_results["data_preprocessing_methods_encode"].unique())
    scale_option = st.sidebar.selectbox("Scaling", df_results["data_preprocessing_methods_scale"].unique())
    correlation_option = st.sidebar.selectbox("Correlation Check", df_results["data_preprocessing_methods_correlation"].unique())
    pca_option = st.sidebar.selectbox("PCA", df_results["data_preprocessing_methods_pca"].unique())
    select_corr_option = st.sidebar.selectbox("Select correlation", df_results["data_preprocessing_methods_select_corr"].unique())

    df_filtered = df_results[
        (df_results['data_preprocessing_methods_outlier'] == outlier_option) &
        (df_results['data_preprocessing_methods_encode'] == encode_option) &
        (df_results['data_preprocessing_methods_scale'] == scale_option) &
        (df_results['data_preprocessing_methods_correlation'] == correlation_option) &
        (df_results['data_preprocessing_methods_pca'] == pca_option) &
        (df_results['data_preprocessing_methods_select_corr'] == select_corr_option)
        
    ]

    if df_filtered.empty:
        st.warning("⚠️ No records match the selected preprocessing setup.")
    else:
        agg_metrics = df_filtered.groupby("model").mean(numeric_only=True).reset_index()

        # --- Bar Chart
        # Accuracy
        
        st.subheader("📊 Validation vs Test Accuracy")
        fig_acc = px.bar(agg_metrics, x="model", y=["valid_accuracy", "test_accuracy"], barmode='group', title="Model Accuracy Comparison")
        st.plotly_chart(fig_acc, use_container_width=True)
        download_plotly_fig(fig_acc, "model_accuracy_comparison.png")

        # Precision
        st.subheader("📊 Validation vs Test Precision")
        fig_prec = px.bar(agg_metrics, x="model", y=["valid_precision", "test_precision"], barmode="group", title="Model Precision Comparison")
        st.plotly_chart(fig_prec, use_container_width=True)
        download_plotly_fig(fig_prec, "model_precision_comparison.png")

        # Recall
        st.subheader("📊 Validation vs Test Recall")
        fig_rec = px.bar(agg_metrics, x="model", y=["valid_recall", "test_recall"], barmode="group", title="Model Recall Comparison")
        st.plotly_chart(fig_rec, use_container_width=True)
        download_plotly_fig(fig_rec, "model_recall_comparison.png")

        # F1 Score
        st.subheader("📊 Validation vs Test F1 Score")
        fig_f1 = px.bar(agg_metrics, x="model", y=["valid_f1_score", "test_f1_score"], barmode="group", title="Model Accuracy Comparison")
        st.plotly_chart(fig_f1, use_container_width=True)
        download_plotly_fig(fig_f1, "model_f1_score_comparison.png")


        # --- Radar Chart
        st.subheader("📈 Radar Chart of Performance Metrics")
        radar_metrics = ["valid_accuracy", "valid_f1_score", "test_accuracy", "test_f1_score"]
        fig_radar = go.Figure()
        for model in agg_metrics["model"]:
            values = agg_metrics[agg_metrics["model"] == model][radar_metrics].values.flatten().tolist()
            fig_radar.add_trace(go.Scatterpolar(r=values, theta=radar_metrics, fill='toself', name=model))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
        st.plotly_chart(fig_radar, use_container_width=True)
        download_plotly_fig(fig_radar, "radar_chart_model_performance.png")

        # # --- Dummy Confusion Matrix
        # st.subheader("🧩 Confusion Matrix (Simulated)")
        # selected_model = st.selectbox("Select Model for Confusion Matrix", agg_metrics["model"])
        # y_true = np.random.randint(0, 2, 100)
        # y_pred = np.random.randint(0, 2, 100)
        # cm = confusion_matrix(y_true, y_pred)
        # fig_cm, ax_cm = plt.subplots()
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        # disp.plot(ax=ax_cm)
        # st.pyplot(fig_cm)
        # download_matplotlib_fig(fig_cm, f"{selected_model}_confusion_matrix.png")

# =============================================
# DASHBOARD 2
if "Dashboard 2" in mode:
    st.header("🔎 Dashboard 2: Preprocessing Pipeline Comparison for One Model")

    selected_model = st.sidebar.selectbox("Select a Model", df_results["model"].unique())
    df_model = df_results[df_results["model"] == selected_model].copy()  # <--- FIXED

    if df_model.empty:
        st.warning("⚠️ No entries found for this model.")
    else:
        df_model["pipeline"] = (
            df_model["data_preprocessing_methods_outlier"].astype(str) + " | " +
            df_model["data_preprocessing_methods_encode"] + " | " +
            df_model["data_preprocessing_methods_scale"] + " | " +
            df_model["data_preprocessing_methods_correlation"].astype(str) + " | " +
            df_model["data_preprocessing_methods_pca"].astype(str) + " | " +
            df_model["data_preprocessing_methods_select_corr"].astype(str) 
        )

        # --- Bar Chart
        st.subheader("📊 Test Accuracy across Pipelines")
        fig_bar2 = px.bar(df_model, x="pipeline", y="test_accuracy", title=f"{selected_model} - Pipeline Performance")
        fig_bar2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar2, use_container_width=True)
        download_plotly_fig(fig_bar2, "bar_pipeline_accuracy.png")

        
        #         # --- Test Accuracy across Pipelines
        # st.subheader("📊 Test Accuracy across Pipelines")
        # fig_bar2 = px.bar(df_model, x="pipeline", y="test_accuracy", 
        #                 title=f"{selected_model} - Pipeline Performance (Accuracy)")
        # fig_bar2.update_layout(xaxis_tickangle=-45)
        # st.plotly_chart(fig_bar2, use_container_width=True)
        # download_plotly_fig(fig_bar2, "bar_pipeline_accuracy.png")


        # --- Test Precision across Pipelines
        st.subheader("📊 Test Precision across Pipelines")
        fig_bar3 = px.bar(df_model, x="pipeline", y="test_precision", 
                        title=f"{selected_model} - Pipeline Performance (Precision)")
        fig_bar3.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar3, use_container_width=True)
        download_plotly_fig(fig_bar3, "bar_pipeline_precision.png")


        # --- Test Recall across Pipelines
        st.subheader("📊 Test Recall across Pipelines")
        fig_bar4 = px.bar(df_model, x="pipeline", y="test_recall", 
                        title=f"{selected_model} - Pipeline Performance (Recall)")
        fig_bar4.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar4, use_container_width=True)
        download_plotly_fig(fig_bar4, "bar_pipeline_recall.png")


        # --- Test F1-Score across Pipelines
        st.subheader("📊 Test F1-Score across Pipelines")
        fig_bar5 = px.bar(df_model, x="pipeline", y="test_f1_score", 
                        title=f"{selected_model} - Pipeline Performance (F1-Score)")
        fig_bar5.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar5, use_container_width=True)
        download_plotly_fig(fig_bar5, "bar_pipeline_f1score.png")





        # --- Radar Chart
        st.subheader("📈 Radar Comparison of Pipelines")
        radar_metrics = ["valid_accuracy", "valid_f1_score", "test_accuracy", "test_f1_score"]
        grouped = df_model.groupby("pipeline").mean(numeric_only=True).reset_index()
        fig_radar2 = go.Figure()
        for _, row in grouped.iterrows():
            fig_radar2.add_trace(go.Scatterpolar(
                r=[row[m] for m in radar_metrics],
                theta=radar_metrics,
                fill='toself',
                name=row["pipeline"]
            ))
        fig_radar2.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
        st.plotly_chart(fig_radar2, use_container_width=True)
        download_plotly_fig(fig_radar2, "radar_pipeline_comparison.png")

        # --- Sankey Diagram
        st.subheader("🔄 Sankey Diagram: Preprocessing Flow")
        outlier_vals = df_model["data_preprocessing_methods_outlier"].unique().tolist()
        encode_vals = df_model["data_preprocessing_methods_encode"].unique().tolist()
        scale_vals = df_model["data_preprocessing_methods_scale"].unique().tolist()

        node_labels = outlier_vals + encode_vals + scale_vals + [selected_model]
        node_count = len(node_labels)

        links_source = []
        links_target = []
        links_value = []

        for o in outlier_vals:
            for e in encode_vals:
                count = df_model[(df_model["data_preprocessing_methods_outlier"] == o) & 
                                 (df_model["data_preprocessing_methods_encode"] == e)].shape[0]
                if count > 0:
                    links_source.append(node_labels.index(o))
                    links_target.append(len(outlier_vals) + encode_vals.index(e))
                    links_value.append(count)

        for e in encode_vals:
            for s in scale_vals:
                count = df_model[(df_model["data_preprocessing_methods_encode"] == e) & 
                                 (df_model["data_preprocessing_methods_scale"] == s)].shape[0]
                if count > 0:
                    links_source.append(len(outlier_vals) + encode_vals.index(e))
                    links_target.append(len(outlier_vals) + len(encode_vals) + scale_vals.index(s))
                    links_value.append(count)

        for s in scale_vals:
            count = df_model[df_model["data_preprocessing_methods_scale"] == s].shape[0]
            if count > 0:
                links_source.append(len(outlier_vals) + len(encode_vals) + scale_vals.index(s))
                links_target.append(node_count - 1)
                links_value.append(count)

        sankey_fig = go.Figure(data=[go.Sankey(
            node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=node_labels),
            link=dict(source=links_source, target=links_target, value=links_value)
        )])
        sankey_fig.update_layout(title_text=f"{selected_model} - Sankey Preprocessing Flow")
        st.plotly_chart(sankey_fig, use_container_width=True)
        download_plotly_fig(sankey_fig, "sankey_pipeline_flow.png")

# =============================================
st.success("✅ Dashboard Loaded Successfully!")
