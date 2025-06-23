

class Messages:
    def __init__(self, logger=None):
        self.logger = logger

    def welcome_message(self):
        # --- DISPLAY FULL PREPROCESSING PIPELINE MESSAGE ---
        print("""
==================================================
    🧠  PD Multi-Modal Prediction Pipeline  🧠
==================================================

Welcome to the **PD-MultiModal-Prediction** framework!

This pipeline is designed to predict **Parkinson’s Disease (PD) progression** by integrating multimodal data—including clinical scores, imaging metrics, and biomarkers. It combines classical regression techniques with **uncertainty quantification** and **explainability** to help researchers and clinicians better understand model decisions and performance.

Whether you're analyzing important features with SHAP, estimating prediction confidence via NGBoost, or conducting feature ablation studies—this framework has you covered!

🔧 Modules Included:
--------------------------------------------------
1️⃣  Data Integration & Preprocessing  
    - Handles multimodal tabular data  
    - Ensures reproducibility with standard pipelines  

2️⃣  Regression Models  
    - Linear Regression, Random Forests, and NGBoost  
    - NGBoost includes NIG uncertainty modeling and regularization  

3️⃣  Explainability & Feature Importance  
    - SHAP-based visualizations for transparency  
    - Insights into model decisions  

4️⃣  Feature Ablation  
    - Systematically removes features  
    - Evaluates performance drops to gauge feature relevance  

5️⃣  Evaluation Pipelines  
    - Supports nested/sequential cross-validation  
    - Hyperparameter tuning via GridSearchCV or Ray  

📤 Output:  
    - Prediction scores with uncertainty intervals  
    - SHAP plots and feature rankings  
    - Logs from ablation studies  

📘 Tools Used:
- scikit-learn  
- SHAP  
- NGBoost with custom distributions  
- Ray (for scalable hyperparameter tuning)

==================================================
            """)

    def running_modules(self):
        if self.logger:
            self.logger.info("🚀  All configurations loaded. Beginning module execution...")
            self.logger.info(" ")
