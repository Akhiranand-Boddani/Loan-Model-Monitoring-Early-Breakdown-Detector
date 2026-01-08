# Loan Model Monitoring & Early Breakdown Detector

## Overview

This project is a complete, production-grade solution for monitoring machine learning models deployed in financial environments, such as credit risk prediction for banks. It detects early signs of model breakdown due to data drift and concept drift, providing actionable insights and human-readable explanations using a local LLM.

## Features

- **Baseline Model Training:** Train models (Logistic Regression, Random Forest, XGBoost) on historical data, compute and save baseline statistics, feature distributions, and performance metrics.
- **Data Drift Detection:** Compare new (production) data to baseline using statistical tests (KS, PSI, chi-square) for numeric and categorical features.
- **Prediction Drift Monitoring:** Track changes in model output distributions and confidence, even before labels are available.
- **Performance Tracking:** If true labels are available, compute rolling accuracy, F1, ROC-AUC, and other metrics to confirm model degradation.
- **Composite Health Scoring:** Merge drift, prediction, and performance signals into a single breakdown score for robust early warning.
- **Human-Readable Reporting:** Generate clear, actionable reports and explanations using a locally hosted LLM (no external API required).
- **Streamlit UI:** Interactive dashboard for uploading new data, visualizing drift, model diagnostics, and explanations.
- **Dockerized Deployment:** Fully containerized for reproducible local deployment.

## Directory Structure

```
loan_drift_monitoring/
├── create_drift_data.py         # Utility for generating synthetic drifted datasets
├── Dockerfile                   # Containerization setup
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── streamlit_app.py             # Main Streamlit dashboard
├── artifacts/                   # Saved models, stats, plots
│   ├── schema.json
│   ├── baseline_models/
│   ├── baseline_plots/
│   ├── baseline_stats/
├── data/                        # Baseline and drift datasets
│   ├── baseline.csv
│   ├── drift_sample_1.csv
│   ├── drift_sample_2.csv
├── loan_monitoring/             # Core Python package
│   ├── config.py                # Global config
│   ├── baseline/                # Baseline stats & model training
│   ├── data_processing/         # Data loading, validation, preprocessing
│   ├── drift/                   # Drift metrics, prediction drift, summary
│   ├── llm/                     # Local LLM explainer
│   ├── utils/                   # Persistence, metrics, scoring
│   ├── visualization/           # Baseline, drift, model plots
```

## Workflow

1. **Baseline Setup:**
   - Train models on baseline data, compute feature and prediction distributions, save stats and models.
2. **Monitoring:**
   - Upload new data via Streamlit UI. Validate schema, compare feature distributions to baseline, compute drift scores.
   - Visualize drift and model output changes.
3. **Prediction Drift:**
   - Analyze changes in model output distributions and confidence scores.
4. **Performance Evaluation:**
   - If labels are present, compute performance metrics and trends.
5. **Composite Breakdown Score:**
   - Merge all signals into a single health score for early warning.
6. **Explanation & Reporting:**
   - Generate human-readable drift explanations using a local LLM.

## Key Modules

- **loan_monitoring/baseline/**: Model training, baseline stats, feature analysis.
- **loan_monitoring/data_processing/**: Data loading, schema validation, preprocessing.
- **loan_monitoring/drift/**: Feature drift, prediction drift, performance evaluation, summary.
- **loan_monitoring/llm/**: Local LLM prompt templates and explainer.
- **loan_monitoring/utils/**: Persistence, metrics, scoring utilities.
- **loan_monitoring/visualization/**: Baseline, drift, and model diagnostic plots.

## Usage

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **(Optional) Build Docker image:**

   ```bash
   docker build -t loan-drift-monitoring .
   docker run -p 8501:8501 loan-drift-monitoring
   ```

3. **Run Streamlit app locally:**

   ```bash
   streamlit run loan_drift_monitoring/streamlit_app.py
   ```

4. **Upload new data:**
   - Use the dashboard to upload CSV files and view drift analysis, model diagnostics, and explanations.

## Local LLM Setup

- The system uses Hugging Face Transformers for local LLM explanations. Ensure you have enough resources to run the selected model (e.g., GPT-J).
- If `transformers` is not installed, install it:
  
   ```bash
   pip install transformers
   ```

## Extending & Customizing

- Add new models or drift metrics in the `baseline` and `drift` modules.
- Customize LLM prompts in `loan_monitoring/llm/prompts.py`.
- Add new visualizations in `loan_monitoring/visualization/`.

## License

MIT License

## Authors

- Project by [Your Name/Team]

## References

- [Evidently AI](https://evidentlyai.com/)
- [Galileo AI](https://www.galileo-ai.com/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Scikit-learn](https://scikit-learn.org/)
- [Streamlit](https://streamlit.io/)
