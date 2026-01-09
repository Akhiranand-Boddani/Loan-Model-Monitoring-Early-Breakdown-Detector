import streamlit as st
import pandas as pd
import numpy as np

# config
from loan_monitoring.config import (
    PAGE_TITLE, TARGET_COLUMN, ID_COLUMN,
    FEATURE_COLUMNS, BASELINE_DATA_FILE,
    BASELINE_STATS_PATH, FEATURE_ANALYSIS_PATH,
    BASELINE_MODEL_METRICS, MODEL_PATHS
)

# data utils
from loan_monitoring.data_processing.load_data import load_csv
from loan_monitoring.data_processing.validation import validate_schema

# persistence
from loan_monitoring.utils.persistence import load_json, load_model

# visualization
from loan_monitoring.visualization.baseline_viz import (
    plot_numeric_histograms, plot_categorical_bars, plot_correlation_heatmap
)
from loan_monitoring.visualization.drift_viz import (
    plot_numeric_drift, plot_categorical_drift
)
from loan_monitoring.visualization.model_viz import (
    plot_roc_curve, plot_confusion_matrix, plot_prediction_distribution
)

# drift
from loan_monitoring.drift.drift_metrics import compute_feature_drift
from loan_monitoring.drift.drift_metrics import compute_conditional_numeric_drift, compute_conditional_categorical_drift
from loan_monitoring.drift.prediction_drift import compute_prediction_drift
from loan_monitoring.drift.performance_eval import compute_performance_metrics
from loan_monitoring.drift.performance_eval import page_hinkley_detector
from loan_monitoring.drift.summary import generate_drift_report
from loan_monitoring.config import THRESHOLDS
from loan_monitoring.utils.metrics import wilson_ci, two_proportion_z_test, bh_adjust, fisher_exact

# local LLM explainer
from loan_monitoring.llm.explainer import explain_drift_locally

st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state="expanded")
st.title("Loan Model Monitoring & Drift Dashboard")
st.markdown("---")

# =========================
# LOAD BASELINE ARTIFACTS
# =========================
@st.cache_data
def load_baseline_data():
    return load_csv(BASELINE_DATA_FILE)

@st.cache_data
def load_stats():
    return load_json(BASELINE_STATS_PATH), load_json(FEATURE_ANALYSIS_PATH)

@st.cache_resource
def load_models():
    return {name: load_model(path) for name, path in MODEL_PATHS.items()}

baseline_df = load_baseline_data()
baseline_stats, feature_analysis = load_stats()
baseline_models = load_models()

# Extract feature types
num_feats = baseline_df.drop(columns=[ID_COLUMN, TARGET_COLUMN]).select_dtypes(include=["number"]).columns.tolist()
cat_feats = baseline_df.drop(columns=[ID_COLUMN, TARGET_COLUMN]).select_dtypes(exclude=["number"]).columns.tolist()

# =========================
# SIDEBAR NAVIGATION
# =========================
with st.sidebar:
    st.markdown("## Navigation")
    st.markdown("**Monitoring Phases:**")
    phase = st.radio(
        "Select Phase",
        options=[
            "Phase 1: Baseline Setup",
            "Phase 2: Input Drift Detection",
            "Phase 3: Prediction Drift",
            "Phase 4: Performance Monitoring",
            "Phase 5: Breakdown Score",
            "Phase 6: Explanations"
        ],
        key="phase_selector"
    )
    
    st.markdown("---")
    st.markdown("### Dataset Info")
    st.metric("Baseline Records", baseline_df.shape[0])
    st.metric("Features", len(FEATURE_COLUMNS))
    st.metric("Default Rate", f"{baseline_df[TARGET_COLUMN].mean():.2%}")

    st.markdown("---")
    with st.expander("Drift Thresholds & Alerts", expanded=False):
        psi_action = st.slider("PSI action threshold", 0.0, 1.0, float(THRESHOLDS['psi']['action']), step=0.01)
        ks_action = st.slider("KS p-value action threshold", 0.0, 0.2, float(THRESHOLDS['ks_pvalue']['action']), step=0.001)
        chi2_action = st.slider("Chi2 p-value action threshold", 0.0, 0.2, float(THRESHOLDS.get('chi2_pvalue', {}).get('action', 0.05)), step=0.001)
        wasser_action = st.slider("Wasserstein action threshold (normalized)", 0.0, 1.0, float(THRESHOLDS['wasserstein']['action']), step=0.01)
        kl_action = st.slider("KL divergence action threshold", 0.0, 1.0, float(THRESHOLDS['kl']['action']), step=0.01)
        st.markdown("Adjust these thresholds to tune sensitivity of drift detection.")

        st.markdown("**Page–Hinkley (CUSUM) Parameters**")
        ph_delta = st.slider("PH delta (sensitivity)", 0.0, 0.1, 0.005, step=0.001)
        ph_lambda = st.slider("PH lambda (threshold)", 0.0, 0.1, 0.02, step=0.001)
        ph_alpha = st.slider("PH alpha (forgetting factor)", 0.0, 1.0, 0.9, step=0.01)
        st.markdown("Adjust PH params to tune how quickly persistent drops are signaled.")
        pd_change_thresh = st.slider("PD change alert threshold (absolute)", 0.0, 1.0, 0.10, step=0.01)

        # Build runtime thresholds dict to pass into drift computations
        runtime_thresholds = {
            "psi": float(psi_action),
            "ks_p_value": float(ks_action),
            "chi2_p_value": float(chi2_action),
            "wasserstein": float(wasser_action),
            "kl_divergence": float(kl_action)
        }
        st.markdown("**Drift Flagging Policy**")
        flag_policy = st.selectbox("Flagging policy", options=["any", "ks_and_effect", "majority"], index=0)
        st.markdown("**Decile PD Test Options**")
        min_samples_per_decile = st.slider("Min samples per decile", 1, 200, 20, step=1)
        pvalue_correction = st.selectbox("P-value correction", options=["none", "bonferroni", "bh"], index=2)
        effect_size_thresh = st.slider("Min absolute PD change to flag", 0.0, 1.0, 0.05, step=0.01)

# =========================
# PHASE 1: BASELINE SETUP
# =========================
if phase == "Phase 1: Baseline Setup":
    st.header("PHASE 1 — Baseline Model & Dataset Characterization")
    st.markdown("**Goal**: Establish a stable training baseline against which future drift is measured.")
    
    # ---- SECTION A: DESCRIPTIVE STATISTICS ----
    st.subheader("Descriptive Statistics")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Numeric Features - Summary**")
        numeric_summary = baseline_df[num_feats].describe().round(3)
        st.dataframe(numeric_summary, use_container_width=True, height=400)
    
    with col2:
        st.markdown("**Target Distribution**")
        target_dist = baseline_df[TARGET_COLUMN].value_counts(normalize=True).round(4)
        st.dataframe(target_dist, use_container_width=True)
    
    st.markdown("---")
    
    # ---- SECTION B: FEATURE DISTRIBUTIONS ----
    st.subheader("Feature Distributions")
    
    tab_numeric, tab_categorical = st.tabs(["Numeric Features", "Categorical Features"])
    
    with tab_numeric:
        st.markdown("**Numeric Feature Histograms**")
        fig1 = plot_numeric_histograms(baseline_df, num_feats)
        if fig1:
            st.pyplot(fig1, use_container_width=True)
    
    with tab_categorical:
        st.markdown("**Categorical Feature Distributions**")
        fig2 = plot_categorical_bars(baseline_df, cat_feats)
        if fig2:
            st.pyplot(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # ---- SECTION C: CORRELATION & FEATURE IMPORTANCE ----
    st.subheader("Correlation & Feature Importance")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Correlation Heatmap (Numeric Features)**")
        fig3 = plot_correlation_heatmap(baseline_df, num_feats)
        if fig3:
            st.pyplot(fig3, use_container_width=True)
    
    with col2:
        st.markdown("**Feature Importance from Tree Models**")
        if feature_analysis and "feature_importance" in feature_analysis:
            fi_df = pd.DataFrame(feature_analysis["feature_importance"], columns=["Feature", "Importance"]).sort_values("Importance", ascending=False).head(10)
            st.dataframe(fi_df, use_container_width=True)
        else:
            st.info("Feature importance data not available.")
    
    st.markdown("---")
    
    # ---- SECTION D: BASELINE MODEL PERFORMANCE ----
    st.subheader("Baseline Model Performance (Offline)")
    
    model_metrics = load_json(BASELINE_MODEL_METRICS)
    
    # Display metrics in columns
    models_to_show = list(model_metrics.keys())
    metrics_cols = st.columns(len(models_to_show))
    
    for idx, (model_name, metrics_cols_item) in enumerate(zip(models_to_show, metrics_cols)):
        with metrics_cols_item:
            st.markdown(f"**{model_name.replace('_', ' ').title()}**")
            metrics = model_metrics[model_name]
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
            st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
            st.metric("F1-Score", f"{metrics.get('f1', 0):.3f}")
            st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
            st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
    
    st.markdown("---")
    
    # ---- SECTION E: MODEL DIAGNOSTIC PLOTS ----
    st.subheader("Model Diagnostic Plots")
    
    for model_name, model in baseline_models.items():
        st.markdown(f"### {model_name.replace('_', ' ').title()}")
        
        try:
            X_baseline = baseline_df.drop(columns=[ID_COLUMN, TARGET_COLUMN])
            y_true = baseline_df[TARGET_COLUMN]
            
            y_proba = model.predict_proba(X_baseline)[:, 1]
            y_pred = model.predict(X_baseline)
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                fig_roc = plot_roc_curve(y_true, y_proba, model_name)
                st.pyplot(fig_roc, use_container_width=True)
            
            with col2:
                fig_cm = plot_confusion_matrix(y_true, y_pred, model_name)
                st.pyplot(fig_cm, use_container_width=True)
            
            with col3:
                fig_pred = plot_prediction_distribution(y_proba, model_name)
                st.pyplot(fig_pred, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error with {model_name}: {str(e)}")
    
    st.success("Phase 1 Complete: Baseline established successfully!")

# =========================
# PHASES 2-6: REQUIRE DATA UPLOAD
# =========================
if phase in ["Phase 2: Input Drift Detection", "Phase 3: Prediction Drift", "Phase 4: Performance Monitoring", "Phase 5: Breakdown Score", "Phase 6: Explanations"]:
    
    st.header(f"{phase}")
    st.markdown("**Upload a new dataset to perform drift analysis.**")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
    
    if not uploaded_file:
        st.info("Please upload a CSV file to proceed.")
        st.stop()
    
    new_df = pd.read_csv(uploaded_file)
    
    # Schema validation
    try:
        validate_schema(
            new_df,
            expected_columns=FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
            id_column=ID_COLUMN
        )
        st.success("Schema matches baseline!")
    except Exception as e:
        st.error(f"❌ Schema error: {e}")
        st.stop()
    
    st.metric("New Data Records", new_df.shape[0])
    
    # =========================
    # PHASE 2: INPUT DRIFT DETECTION
    # =========================
    if phase == "Phase 2: Input Drift Detection":
        st.markdown("---")
        st.markdown("**Goal**: Detect changes in input features over time before labels arrive.")
        st.markdown("**Tests Used**: KS (numeric), Chi-Square (categorical), PSI, Wasserstein, KL Divergence")
        
        st.subheader("Comprehensive Drift Analysis")
        
        feature_drift = compute_feature_drift(
            baseline_stats, new_df, ignore_columns=[ID_COLUMN, TARGET_COLUMN], thresholds=runtime_thresholds, flagging_policy=flag_policy
        )
        
        # Separate numeric and categorical
        numeric_drift = {k: v for k, v in feature_drift.items() if v.get("feature_type") == "numeric"}
        categorical_drift = {k: v for k, v in feature_drift.items() if v.get("feature_type") == "categorical"}
        
        # ---- NUMERIC DRIFT TABLE ----
        if numeric_drift:
            st.markdown("### Numeric Features Drift")
            
            numeric_table = []
            for feature, results in numeric_drift.items():
                metrics = results.get("metrics", {})
                numeric_table.append({
                    "Feature": feature,
                    "PSI": f"{metrics.get('psi', 0):.4f}",
                    "KS Statistic": f"{metrics.get('ks_statistic', 0):.4f}",
                    "KS P-Value": f"{metrics.get('ks_p_value', 0):.6f}",
                    "Wasserstein": f"{metrics.get('wasserstein_normalized', 0):.4f}",
                    "KL Divergence": f"{metrics.get('kl_divergence', 0):.4f}",
                    "Status": "DRIFT" if results.get("is_drift", False) else "OK"
                })
            
            numeric_df = pd.DataFrame(numeric_table)
            st.dataframe(numeric_df, use_container_width=True, height=300)
            
            # Show detailed metrics for drifted features
            drifted_numeric = {k: v for k, v in numeric_drift.items() if v.get("is_drift")}
            if drifted_numeric:
                st.markdown("**Drifted Numeric Features - Detailed Stats:**")
                for feat, results in drifted_numeric.items():
                    with st.expander(f"{feat} - Detailed Analysis"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Baseline Stats**")
                            base_stats = results.get("baseline_stats", {})
                            st.metric("Mean", f"{base_stats.get('mean', 0):.4f}")
                            st.metric("Std Dev", f"{base_stats.get('std', 0):.4f}")
                        with col2:
                            st.markdown("**New Data Stats**")
                            new_stats = results.get("new_stats", {})
                            st.metric("Mean", f"{new_stats.get('mean', 0):.4f}")
                            st.metric("Std Dev", f"{new_stats.get('std', 0):.4f}")
            
            st.markdown("---")

        # ---- CONDITIONAL (CLASS-CONDITIONED) DRIFT ----
        if TARGET_COLUMN in new_df.columns:
            st.markdown("### Conditional (Class-Conditioned) Drift Analysis")
            st.markdown("This checks feature shifts conditioned on the target (helps detect concept drift).")
            # numeric conditional drift
            cond_numeric_table = []
            for feat in num_feats:
                cond = compute_conditional_numeric_drift(baseline_df, new_df, feat, TARGET_COLUMN)
                # show PSI for class=1 (defaults) and class=0
                psi_1 = cond.get(1, {}).get('psi') if 1 in cond else cond.get('Y', {}).get('psi')
                psi_0 = cond.get(0, {}).get('psi') if 0 in cond else None
                cond_numeric_table.append({"Feature": feat, "PSI_Y1": psi_1, "PSI_Y0": psi_0})
            st.dataframe(pd.DataFrame(cond_numeric_table), use_container_width=True)

            # categorical conditional drift
            cond_cat_table = []
            for feat in cat_feats:
                condc = compute_conditional_categorical_drift(baseline_df, new_df, feat, TARGET_COLUMN)
                chi2_y1 = condc.get(1, {}).get('chi2_p') if 1 in condc else None
                chi2_y0 = condc.get(0, {}).get('chi2_p') if 0 in condc else None
                cond_cat_table.append({"Feature": feat, "Chi2P_Y1": chi2_y1, "Chi2P_Y0": chi2_y0})
            st.dataframe(pd.DataFrame(cond_cat_table), use_container_width=True)

            st.markdown("---")

            # ---- PD DECILE COMPARISON & CUSUM ON RECALL ----
            st.subheader("Decile PD Comparison & Recall CUSUM")
            model_choice = st.selectbox("Model for PD / CUSUM", options=list(baseline_models.keys()), index=2)
            model = baseline_models[model_choice]

            # compute probabilities
            X_base = baseline_df.drop(columns=[ID_COLUMN, TARGET_COLUMN])
            X_new = new_df.drop(columns=[ID_COLUMN, TARGET_COLUMN])
            y_base = baseline_df[TARGET_COLUMN]
            y_new = new_df[TARGET_COLUMN]
            try:
                proba_base = model.predict_proba(X_base)[:, 1]
                proba_new = model.predict_proba(X_new)[:, 1]
            except Exception as e:
                st.warning(f"Model predict_proba failed: {e}")
                proba_base = None
                proba_new = None

            if proba_base is not None:
                base_df_pd = pd.DataFrame({"proba": proba_base, "target": y_base})
                new_df_pd = pd.DataFrame({"proba": proba_new, "target": y_new})

                # Derive decile edges from baseline and apply to both datasets
                try:
                    edges = np.quantile(proba_base, np.linspace(0, 1, 11))
                    edges = np.unique(edges)
                    if len(edges) < 2:
                        raise ValueError("Not enough variability in baseline scores")
                except Exception:
                    st.info("Not enough variability in baseline scores to form deciles.")
                    edges = None

                if edges is not None and len(edges) >= 2:
                    # create decile bins using baseline edges
                    base_df_pd["decile"] = pd.cut(base_df_pd["proba"], bins=edges, include_lowest=True, labels=False)
                    new_df_pd["decile"] = pd.cut(new_df_pd["proba"], bins=edges, include_lowest=True, labels=False)

                    # build stats per decile
                    deciles = sorted(pd.unique(base_df_pd["decile"].dropna().astype(int).tolist()))
                    rows = []
                    pvals = []
                    for d in deciles:
                        base_slice = base_df_pd[base_df_pd["decile"] == d]
                        new_slice = new_df_pd[new_df_pd["decile"] == d]
                        b_n = int(len(base_slice))
                        n_n = int(len(new_slice))
                        b_pos = int(base_slice["target"].sum()) if b_n > 0 else 0
                        n_pos = int(new_slice["target"].sum()) if n_n > 0 else 0
                        b_pd = b_pos / b_n if b_n > 0 else 0.0
                        n_pd = n_pos / n_n if n_n > 0 else 0.0
                        b_ci_low, b_ci_high = wilson_ci(b_pos, b_n)
                        n_ci_low, n_ci_high = wilson_ci(n_pos, n_n)
                        abs_change = n_pd - b_pd

                        # decide test: Fisher for small counts, else z-test
                        use_fisher = (b_n < 30 or n_n < 30)
                        pval = None
                        zstat = None
                        if b_n >= 1 and n_n >= 1 and b_n >= min_samples_per_decile and n_n >= min_samples_per_decile:
                            if use_fisher:
                                try:
                                    table = [[b_pos, b_n - b_pos], [n_pos, n_n - n_pos]]
                                    odds, pval = fisher_exact(table)
                                except Exception:
                                    pval = 1.0
                            else:
                                pval, zstat = two_proportion_z_test(b_pos, b_n, n_pos, n_n)
                        else:
                            pval = np.nan

                        rows.append({
                            "decile": int(d),
                            "base_n": b_n,
                            "base_pos": b_pos,
                            "base_pd": b_pd,
                            "base_ci_low": b_ci_low,
                            "base_ci_high": b_ci_high,
                            "new_n": n_n,
                            "new_pos": n_pos,
                            "new_pd": n_pd,
                            "new_ci_low": n_ci_low,
                            "new_ci_high": n_ci_high,
                            "abs_change": abs_change,
                            "p_value": pval
                        })
                        pvals.append(pval if pval is not None else np.nan)

                    dec_df = pd.DataFrame(rows).set_index("decile")

                    # p-value correction
                    valid_pidx = ~dec_df["p_value"].isna()
                    pvals_list = dec_df.loc[valid_pidx, "p_value"].tolist()
                    if pvalue_correction == "bonferroni":
                        adj = [min(p * len(pvals_list), 1.0) for p in pvals_list]
                    elif pvalue_correction == "bh":
                        adj = bh_adjust(pvals_list)
                    else:
                        adj = pvals_list
                    # write back adjusted
                    dec_df.loc[valid_pidx, "p_adj"] = adj
                    dec_df["p_adj"] = dec_df["p_adj"].fillna(np.nan)

                    # determine flags: require effect size AND p-value
                    dec_df["flag"] = False
                    for idx, row in dec_df.iterrows():
                        p_adj = row.get("p_adj")
                        if np.isnan(p_adj):
                            dec_df.at[idx, "flag"] = False
                        else:
                            dec_df.at[idx, "flag"] = (abs(row["abs_change"]) >= effect_size_thresh) and (p_adj < ks_action)

                    # show table
                    st.dataframe(dec_df.reset_index(), use_container_width=True)

                    # plot bar chart with error bars
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 4))
                    indices = np.arange(len(dec_df))
                    width = 0.35
                    ax.bar(indices - width/2, dec_df["base_pd"], width, yerr=[dec_df["base_pd"] - dec_df["base_ci_low"], dec_df["base_ci_high"] - dec_df["base_pd"]], label="baseline_PD", alpha=0.6)
                    ax.bar(indices + width/2, dec_df["new_pd"], width, yerr=[dec_df["new_pd"] - dec_df["new_ci_low"], dec_df["new_ci_high"] - dec_df["new_pd"]], label="new_PD", alpha=0.8)
                    ax.set_xticks(indices)
                    ax.set_xticklabels(dec_df.index.tolist())
                    ax.set_xlabel("Decile")
                    ax.set_ylabel("PD")
                    ax.legend()
                    st.pyplot(fig, use_container_width=True)

                    # show recommendations if flags
                    flagged = dec_df[dec_df["flag"]]
                    if not flagged.empty:
                        st.warning(f"{len(flagged)} decile(s) show significant PD change. See recommendations below.")
                        st.markdown("**Recommended actions:**\n- Prioritize labeling for flagged deciles.\n- Retrain model including recent flagged decile samples.\n- Investigate feature shifts within flagged deciles.")
                else:
                    st.info("Not enough variability to form deciles; skipping PD decile comparison.")

                # CUSUM on recall across chunks
                if len(new_df) > 100 and TARGET_COLUMN in new_df.columns:
                    from sklearn.metrics import recall_score
                    chunks = np.array_split(new_df, 20)
                    recalls = []
                    for c in chunks:
                        Xc = c.drop(columns=[ID_COLUMN, TARGET_COLUMN])
                        yc = c[TARGET_COLUMN]
                        try:
                            yp = model.predict(Xc)
                            r = recall_score(yc, yp)
                        except Exception:
                            r = 0.0
                        recalls.append(r)
                    recall_series = pd.Series(recalls)
                    # feed into PH detector to check sustained drops using sidebar params
                    ph_inst = page_hinkley_detector(delta=ph_delta, lambda_=ph_lambda, alpha=ph_alpha)
                    signals = [ph_inst.update(float(v)) for v in recall_series]
                    sig_int = [int(s) for s in signals]
                    st.line_chart(pd.DataFrame({"recall": recall_series, "signal": sig_int}))
                    if any(signals):
                        st.error("Persistent recall degradation detected by Page–Hinkley detector. Investigate immediately.")
        
        # ---- CATEGORICAL DRIFT TABLE ----
        if categorical_drift:
            st.markdown("### Categorical Features Drift")
            
            categorical_table = []
            for feature, results in categorical_drift.items():
                metrics = results.get("metrics", {})
                categorical_table.append({
                    "Feature": feature,
                    "Chi-Square": f"{metrics.get('chi2_statistic', 0):.4f}",
                    "Chi2 P-Value": f"{metrics.get('chi2_p_value', 0):.6f}",
                    "KL Divergence": f"{metrics.get('kl_divergence', 0):.4f}",
                    "Entropy Change": f"{metrics.get('entropy_new', 0) - metrics.get('entropy_baseline', 0):.4f}",
                    "Status": "DRIFT" if results.get("is_drift", False) else "OK"
                })
            
            categorical_df = pd.DataFrame(categorical_table)
            st.dataframe(categorical_df, use_container_width=True, height=300)
            
            # Show detailed metrics for drifted features
            drifted_categorical = {k: v for k, v in categorical_drift.items() if v.get("is_drift")}
            if drifted_categorical:
                st.markdown("**Drifted Categorical Features - Distribution Changes:**")
                for feat, results in drifted_categorical.items():
                    with st.expander(f"{feat} - Distribution Details"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Baseline Distribution**")
                            st.json(results.get("baseline_proportions", {}))
                        with col2:
                            st.markdown("**New Data Distribution**")
                            st.json(results.get("new_proportions", {}))
            
            st.markdown("---")
        
        # ---- INTERPRETATION GUIDE ----
        with st.expander("📖 How to Interpret These Metrics"):
            st.markdown("""
            **PSI (Population Stability Index)**
            - `< 0.10`: No drift
            - `0.10 - 0.25`: Small drift (monitor)
            - `> 0.25`: Significant drift (action needed)
            
            **KS Test (Kolmogorov-Smirnov)**
            - Tests if two distributions are different
            - P-value < 0.05: Reject null hypothesis (distributions differ)
            - More sensitive to central differences than tails
            
            **Chi-Square Test (Categorical)**
            - Tests independence between baseline and new distributions
            - P-value < 0.05: Distributions are significantly different
            
            **Wasserstein Distance**
            - Measures the minimum effort to transform one distribution to another
            - Range: [0, ∞), normalized by feature range
            - Higher = more different
            
            **KL Divergence**
            - Measures how much one distribution differs from expected
            - `0`: Identical | `< 0.1`: Minor | `0.1-0.25`: Moderate | `> 0.25`: Significant
            - Asymmetric: KL(P||Q) ≠ KL(Q||P)
            """)
        
        st.markdown("---")
        st.subheader("Visual Drift Comparison")
        
        tab_num, tab_cat = st.tabs(["Numeric Distributions", "Categorical Distributions"])
        
        with tab_num:
            st.markdown("**Side-by-side histograms of baseline vs new data**")
            fig_drift_num = plot_numeric_drift(baseline_df, new_df, num_feats)
            if fig_drift_num:
                st.pyplot(fig_drift_num, use_container_width=True)
        
        with tab_cat:
            st.markdown("**Side-by-side bar charts of baseline vs new data**")
            fig_drift_cat = plot_categorical_drift(baseline_df, new_df, cat_feats)
            if fig_drift_cat:
                st.pyplot(fig_drift_cat, use_container_width=True)
        
        st.success("Phase 2 Complete: Comprehensive drift analysis done!")
    
    # =========================
    # PHASE 3: PREDICTION DRIFT
    # =========================
    elif phase == "Phase 3: Prediction Drift":
        st.markdown("---")
        st.markdown("**Goal**: Monitor changes in the model's own outputs over time.")
        
        X_new = new_df.drop(columns=[ID_COLUMN, TARGET_COLUMN])
        
        st.subheader("Prediction Distribution Changes")
        
        pred_cols = st.columns(len(baseline_models))
        
        for idx, (model_name, model) in enumerate(baseline_models.items()):
            with pred_cols[idx]:
                y_proba_base = model.predict_proba(baseline_df.drop(columns=[ID_COLUMN, TARGET_COLUMN]))[:, 1]
                y_proba_new = model.predict_proba(X_new)[:, 1]
                
                st.markdown(f"**{model_name.replace('_', ' ').title()}**")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.metric("Baseline Entropy", f"{-np.mean([p * np.log(p + 1e-6) + (1-p) * np.log(1-p + 1e-6) for p in y_proba_base]):.4f}")
                
                with col2:
                    st.metric("New Data Entropy", f"{-np.mean([p * np.log(p + 1e-6) + (1-p) * np.log(1-p + 1e-6) for p in y_proba_new]):.4f}")
        
        st.markdown("---")
        st.subheader("Prediction Distribution Plots")
        
        pred_plot_cols = st.columns(len(baseline_models))
        for idx, (model_name, model) in enumerate(baseline_models.items()):
            with pred_plot_cols[idx]:
                X_new = new_df.drop(columns=[ID_COLUMN, TARGET_COLUMN])
                y_proba_new = model.predict_proba(X_new)[:, 1]
                fig_pred = plot_prediction_distribution(y_proba_new, model_name)
                st.pyplot(fig_pred, use_container_width=True)
        
        st.success("Phase 3 Complete: Prediction drift analysis done!")
    
    # =========================
    # PHASE 4: PERFORMANCE MONITORING
    # =========================
    elif phase == "Phase 4: Performance Monitoring":
        st.markdown("---")
        st.markdown("**Goal**: Track true performance once ground truth labels become available.")
        
        if TARGET_COLUMN not in new_df.columns:
            st.warning("No target column in new data. Cannot compute performance metrics.")
            st.info("Please upload data that includes the target column for performance monitoring.")
            st.stop()
        
        st.subheader("Performance Metrics on New Data")
        
        perf_results = {}
        X_new = new_df.drop(columns=[ID_COLUMN, TARGET_COLUMN])
        y_true_new = new_df[TARGET_COLUMN]
        
        perf_cols = st.columns(len(baseline_models))
        
        for idx, (model_name, model) in enumerate(baseline_models.items()):
            with perf_cols[idx]:
                y_pred_new = model.predict(X_new)
                y_proba_new = model.predict_proba(X_new)[:, 1]
                
                perf_metrics = compute_performance_metrics(y_true_new, y_pred_new, y_proba_new)
                perf_results[model_name] = perf_metrics
                
                st.markdown(f"**{model_name.replace('_', ' ').title()}**")
                st.metric("Accuracy", f"{perf_metrics.get('accuracy', 0):.3f}")
                st.metric("ROC-AUC", f"{perf_metrics.get('roc_auc', 0):.3f}")
                st.metric("F1-Score", f"{perf_metrics.get('f1', 0):.3f}")
                st.metric("Precision", f"{perf_metrics.get('precision', 0):.3f}")
                st.metric("Recall", f"{perf_metrics.get('recall', 0):.3f}")
        
        st.success("Phase 4 Complete: Performance monitoring done!")
    
    # =========================
    # PHASE 5: BREAKDOWN SCORE
    # =========================
    elif phase == "Phase 5: Breakdown Score":
        st.markdown("---")
        st.markdown("**Goal**: Merge multiple signals into an interpretable health score.")
        
        # Compute all drift metrics
        feature_drift = compute_feature_drift(
            baseline_stats, new_df, ignore_columns=[ID_COLUMN, TARGET_COLUMN], thresholds=runtime_thresholds, flagging_policy=flag_policy
        )
        X_new = new_df.drop(columns=[ID_COLUMN, TARGET_COLUMN])
        
        # Simple breakdown score calculation
        drift_count = sum(1 for f in feature_drift.values() if f.get("is_drift", False))
        drift_ratio = drift_count / len(feature_drift) if feature_drift else 0
        
        # Health score: 100 - (drift_ratio * 50) - (performance_drop * 50)
        health_score = 100 * (1 - drift_ratio)
        
        st.subheader("Model Breakdown Health Score")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            color = "OK" if health_score > 70 else "WARNING" if health_score > 40 else "CRITICAL"
            st.metric("Health Score", f"{health_score:.1f}%", delta=f"{color}")
        
        with col2:
            st.metric("Features Drifted", f"{drift_count}/{len(feature_drift)}")
        
        with col3:
            alert_status = "NORMAL" if health_score > 70 else "WARNING" if health_score > 40 else "BREAKDOWN LIKELY"
            st.metric("Alert Status", alert_status)
        
        st.markdown("---")
        
        # Breakdown score gauge visualization
        st.subheader("Score Components")
        
        components = pd.DataFrame({
            "Component": ["Data Drift", "Features Healthy"],
            "Score": [drift_ratio * 100, (1 - drift_ratio) * 100]
        })
        
        st.bar_chart(components.set_index("Component"), use_container_width=True)

        if health_score <= 40:
            st.error("**BREAKDOWN LIKELY**: Model performance severely compromised. Immediate retraining recommended!")
        elif health_score <= 70:
            st.warning("WARNING: Model health declining. Monitor closely.")
        else:
            st.success("NORMAL: Model operating within acceptable parameters.")
        
        st.success("Phase 5 Complete: Breakdown score calculated!")
    
    # =========================
    # PHASE 6: EXPLANATIONS
    # =========================
    elif phase == "Phase 6: Explanations":
        st.markdown("---")
        st.markdown("**Goal**: Use LLM & dashboards to explain what happened and recommend actions.")
        
        # Compute all summaries
        feature_drift = compute_feature_drift(
            baseline_stats, new_df, ignore_columns=[ID_COLUMN, TARGET_COLUMN], thresholds=runtime_thresholds, flagging_policy=flag_policy
        )
        X_new = new_df.drop(columns=[ID_COLUMN, TARGET_COLUMN])
        pred_drift = compute_prediction_drift(X_new, MODEL_PATHS)
        
        perf_results = None
        if TARGET_COLUMN in new_df.columns:
            y_true_new = new_df[TARGET_COLUMN]
            perf_results = {}
            for model_name, model in baseline_models.items():
                y_pred_new = model.predict(X_new)
                y_proba_new = model.predict_proba(X_new)[:, 1]
                perf_results[model_name] = compute_performance_metrics(y_true_new, y_pred_new, y_proba_new)
        
        st.subheader("AI-Generated Drift Explanation")
        
        try:
            explanation = explain_drift_locally(
                feature_summary=feature_drift,
                prediction_summary=pred_drift,
                performance_summary=perf_results
            )
            st.info(explanation)
        except Exception as llm_err:
            st.warning(f"LLM explanation not available: {llm_err}")
            st.markdown("**Summary of Key Findings:**")
            
            # Fallback text summary
            drift_features = [f for f, r in feature_drift.items() if r.get("is_drift", False)]
            if drift_features:
                st.markdown(f"- **{len(drift_features)} features showed drift**: {', '.join(drift_features[:5])}...")
            else:
                st.markdown("- No significant feature drift detected")
            
            if perf_results:
                st.markdown("- Performance metrics computed successfully")
            else:
                st.markdown("- No target labels provided; cannot assess performance change")
        
        # -----------------------------
        # Prediction-conditioned drift + recommendations
        # -----------------------------
        st.markdown("---")
        st.subheader("🔎 Prediction-Conditioned Drift & Recommendations")
        st.markdown("Compare baseline vs new predicted default rates per score-decile and suggest actions.")

        for model_name, model in baseline_models.items():
            st.markdown(f"**Model: {model_name.replace('_', ' ').title()}**")
            try:
                X_base = baseline_df.drop(columns=[ID_COLUMN, TARGET_COLUMN])
                y_base = baseline_df[TARGET_COLUMN]
                X_new = new_df.drop(columns=[ID_COLUMN, TARGET_COLUMN])
                y_new = new_df[TARGET_COLUMN] if TARGET_COLUMN in new_df.columns else None

                proba_base = model.predict_proba(X_base)[:, 1]
                proba_new = model.predict_proba(X_new)[:, 1]

                # derive decile edges from baseline
                edges = np.unique(np.quantile(proba_base, np.linspace(0, 1, 11)))
                if len(edges) < 2:
                    st.info("Not enough variability in baseline scores to form deciles.")
                    continue

                base_bins = pd.cut(proba_base, bins=edges, include_lowest=True, labels=False)
                new_bins = pd.cut(proba_new, bins=edges, include_lowest=True, labels=False)

                base_df_pd = pd.DataFrame({"decile": base_bins, "target": y_base, "proba": proba_base})
                new_df_pd = pd.DataFrame({"decile": new_bins, "proba": proba_new})
                if y_new is not None:
                    new_df_pd["target"] = y_new.values

                dec_base = base_df_pd.groupby("decile").target.mean()
                dec_new = new_df_pd.groupby("decile").target.mean()

                comp = pd.DataFrame({"baseline_PD": dec_base, "new_PD": dec_new}).fillna(0)
                comp["abs_change"] = (comp["new_PD"] - comp["baseline_PD"]).abs()
                comp["flag"] = comp["abs_change"] > pd_change_thresh

                st.dataframe(comp.reset_index().rename(columns={"decile": "Decile"}), use_container_width=True)

                flagged = comp[comp["flag"]]
                if not flagged.empty:
                    st.warning(f"Segments with PD change > {pd_change_thresh:.2f} detected: {len(flagged)} decile(s).")
                    st.markdown("**Recommended Actions:**")
                    st.write("- Collect labels for affected segments and prioritize labeling.")
                    st.write("- Retrain model including recent samples from flagged deciles.")
                    st.write("- Investigate feature shifts within flagged score buckets (run conditional drift checks).")
                else:
                    st.success("No major PD changes detected across deciles.")

            except Exception as e:
                st.error(f"Error computing prediction-conditioned drift for {model_name}: {e}")
        
        st.success("Phase 6 Complete: Explanation generated!")


