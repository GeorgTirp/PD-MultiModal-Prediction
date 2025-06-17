# 🧠 PD Multi-Modal Prediction

**PD-MultiModal-Prediction** is a Python framework for predicting Parkinson’s Disease progression using multimodal data (clinical scores, imaging, biomarkers). It integrates classical regression models with uncertainty quantification and model explainability (via SHAP and NGBoost), and supports feature ablation experiments.

---

## 📦 Features

- **Modular regression models**  
  - `BaseRegressionModel` covers preprocessing, cross-validation, SHAP, uncertainty, and ablation.
  - Includes `LinearRegressionModel`, `RandomForestModel`, and NGBoost with a Normal‑Inverse‑Gamma distribution.

- **Uncertainty-aware predictions**  
  - Leverages NGBoost with custom NIG distribution and KL/evidential regularization.

- **Explainability with SHAP**  
  - Provides feature importance scoring and interpretation visualizations.

- **Feature ablation analysis**  
  - Iteratively removes least important features and logs performance impact.

- **Robust evaluation pipelines**  
  - Supports sequential and nested CV, hyperparameter tuning (via `GridSearchCV` or Ray for NGBoost).

---

## 🛠️ Installation

```bash
git clone https://github.com/GeorgTirp/PD-MultiModal-Prediction.git
cd PD-MultiModal-Prediction

# Create and activate your Python environment
conda create -n pdmm python=3.10 pip -y
conda activate pdmm

# Install dependencies
pip install -r requirements.txt
