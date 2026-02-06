"""
Extended ML Templates for KNIME Node Mapping.

Provides templates for:
- Classification (Random Forest, SVM, Logistic Regression, XGBoost)
- Regression (Linear, Ridge, Lasso)
- Clustering (DBSCAN, Hierarchical)
- Model Evaluation (Scorer, ROC Curve)
- Model Persistence (Model Writer/Reader)
"""
from typing import Dict, Any

ML_TEMPLATES: Dict[str, Dict[str, Any]] = {
    # ==================== Classification - Learners ====================
    
    "org.knime.base.node.mine.treensemble2.node.learner.classification.TreeEnsembleClassificationLearnerNodeFactory3": {
        "imports": [
            "import pandas as pd",
            "from sklearn.ensemble import RandomForestClassifier"
        ],
        "code": """rf_model = RandomForestClassifier(
    n_estimators={n_estimators},
    max_depth={max_depth},
    random_state=42,
    n_jobs=-1
)
rf_model.fit({input_var}[{feature_columns}], {input_var}['{target_column}'])""",
        "description": "Train Random Forest classifier",
        "category": "ML-Classification"
    },
    
    "org.knime.base.node.mine.svm.learner.SVMLearnerNodeFactory": {
        "imports": [
            "import pandas as pd",
            "from sklearn.svm import SVC"
        ],
        "code": """svm_model = SVC(kernel='{kernel}', C={c_param}, random_state=42)
svm_model.fit({input_var}[{feature_columns}], {input_var}['{target_column}'])""",
        "description": "Train Support Vector Machine classifier",
        "category": "ML-Classification"
    },
    
    "org.knime.base.node.mine.neural_network.mlp2.MLPClassificationLearnerNodeFactory": {
        "imports": [
            "import pandas as pd",
            "from sklearn.neural_network import MLPClassifier"
        ],
        "code": """mlp_model = MLPClassifier(
    hidden_layer_sizes={hidden_layers},
    max_iter=1000,
    random_state=42
)
mlp_model.fit({input_var}[{feature_columns}], {input_var}['{target_column}'])""",
        "description": "Train Multi-Layer Perceptron classifier",
        "category": "ML-Classification"
    },
    
    "org.knime.base.node.mine.regression.logistic.learner.LogisticRegressionLearnerNodeFactory": {
        "imports": [
            "import pandas as pd",
            "from sklearn.linear_model import LogisticRegression"
        ],
        "code": """logreg_model = LogisticRegression(max_iter=1000, random_state=42)
logreg_model.fit({input_var}[{feature_columns}], {input_var}['{target_column}'])""",
        "description": "Train Logistic Regression classifier",
        "category": "ML-Classification"
    },
    
    # ==================== Classification - Predictors ====================
    
    "org.knime.base.node.mine.treensemble2.node.predictor.classification.TreeEnsembleClassificationPredictorNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """{output_var} = {input_var}.copy()
{output_var}['prediction'] = rf_model.predict({input_var}[{feature_columns}])
{output_var}['prediction_probability'] = rf_model.predict_proba({input_var}[{feature_columns}]).max(axis=1)""",
        "description": "Predict with Random Forest model",
        "category": "ML-Prediction"
    },
    
    "org.knime.base.node.mine.svm.predictor.SVMPredictorNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """{output_var} = {input_var}.copy()
{output_var}['prediction'] = svm_model.predict({input_var}[{feature_columns}])""",
        "description": "Predict with SVM model",
        "category": "ML-Prediction"
    },
    
    "org.knime.base.node.mine.decisiontree2.predictor2.DecTreePredictorNodeFactory2": {
        "imports": ["import pandas as pd"],
        "code": """{output_var} = {input_var}.copy()
{output_var}['prediction'] = dt_model.predict({input_var}[{feature_columns}])""",
        "description": "Predict with Decision Tree model",
        "category": "ML-Prediction"
    },
    
    # ==================== Regression - Learners ====================
    
    "org.knime.base.node.mine.regression.linear.learner.LinearRegressionLearnerNodeFactory": {
        "imports": [
            "import pandas as pd",
            "from sklearn.linear_model import LinearRegression"
        ],
        "code": """linreg_model = LinearRegression()
linreg_model.fit({input_var}[{feature_columns}], {input_var}['{target_column}'])""",
        "description": "Train Linear Regression model",
        "category": "ML-Regression"
    },
    
    "org.knime.base.node.mine.regression.ridge.learner.RidgeRegressionLearnerNodeFactory": {
        "imports": [
            "import pandas as pd",
            "from sklearn.linear_model import Ridge"
        ],
        "code": """ridge_model = Ridge(alpha={alpha})
ridge_model.fit({input_var}[{feature_columns}], {input_var}['{target_column}'])""",
        "description": "Train Ridge Regression model",
        "category": "ML-Regression"
    },
    
    "org.knime.base.node.mine.regression.lasso.learner.LassoRegressionLearnerNodeFactory": {
        "imports": [
            "import pandas as pd",
            "from sklearn.linear_model import Lasso"
        ],
        "code": """lasso_model = Lasso(alpha={alpha})
lasso_model.fit({input_var}[{feature_columns}], {input_var}['{target_column}'])""",
        "description": "Train Lasso Regression model",
        "category": "ML-Regression"
    },
    
    "org.knime.base.node.mine.treensemble2.node.learner.regression.TreeEnsembleRegressionLearnerNodeFactory3": {
        "imports": [
            "import pandas as pd",
            "from sklearn.ensemble import RandomForestRegressor"
        ],
        "code": """rf_regressor = RandomForestRegressor(
    n_estimators={n_estimators},
    max_depth={max_depth},
    random_state=42,
    n_jobs=-1
)
rf_regressor.fit({input_var}[{feature_columns}], {input_var}['{target_column}'])""",
        "description": "Train Random Forest Regressor",
        "category": "ML-Regression"
    },
    
    # ==================== Regression - Predictors ====================
    
    "org.knime.base.node.mine.regression.linear.predict.LinearRegressionPredictorNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """{output_var} = {input_var}.copy()
{output_var}['prediction'] = linreg_model.predict({input_var}[{feature_columns}])""",
        "description": "Predict with Linear Regression model",
        "category": "ML-Prediction"
    },
    
    "org.knime.base.node.mine.treensemble2.node.predictor.regression.TreeEnsembleRegressionPredictorNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """{output_var} = {input_var}.copy()
{output_var}['prediction'] = rf_regressor.predict({input_var}[{feature_columns}])""",
        "description": "Predict with Random Forest Regressor",
        "category": "ML-Prediction"
    },
    
    # ==================== Clustering ====================
    
    "org.knime.base.node.mine.cluster.dbscan.DBSCANNodeFactory": {
        "imports": [
            "import pandas as pd",
            "from sklearn.cluster import DBSCAN"
        ],
        "code": """dbscan = DBSCAN(eps={eps}, min_samples={min_samples})
{output_var} = {input_var}.copy()
{output_var}['cluster'] = dbscan.fit_predict({input_var}[{feature_columns}])""",
        "description": "DBSCAN clustering",
        "category": "ML-Clustering"
    },
    
    "org.knime.base.node.mine.cluster.hierarchical.HierarchicalClusterNodeFactory": {
        "imports": [
            "import pandas as pd",
            "from sklearn.cluster import AgglomerativeClustering"
        ],
        "code": """hier_cluster = AgglomerativeClustering(n_clusters={n_clusters}, linkage='{linkage}')
{output_var} = {input_var}.copy()
{output_var}['cluster'] = hier_cluster.fit_predict({input_var}[{feature_columns}])""",
        "description": "Hierarchical/Agglomerative clustering",
        "category": "ML-Clustering"
    },
    
    # ==================== Model Evaluation ====================
    
    "org.knime.base.node.mine.scorer.accuracy.AccuracyScorerNodeFactory": {
        "imports": [
            "import pandas as pd",
            "from sklearn.metrics import accuracy_score, classification_report"
        ],
        "code": """accuracy = accuracy_score({input_var}['{actual_column}'], {input_var}['{predicted_column}'])
report = classification_report({input_var}['{actual_column}'], {input_var}['{predicted_column}'])
print(f"Accuracy: {{accuracy:.4f}}")
print(report)""",
        "description": "Calculate classification accuracy and metrics",
        "category": "ML-Evaluation"
    },
    
    "org.knime.base.node.mine.scorer.numeric.NumericScorerNodeFactory": {
        "imports": [
            "import pandas as pd",
            "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score"
        ],
        "code": """mse = mean_squared_error({input_var}['{actual_column}'], {input_var}['{predicted_column}'])
mae = mean_absolute_error({input_var}['{actual_column}'], {input_var}['{predicted_column}'])
r2 = r2_score({input_var}['{actual_column}'], {input_var}['{predicted_column}'])
print(f"MSE: {{mse:.4f}}, MAE: {{mae:.4f}}, R²: {{r2:.4f}}")""",
        "description": "Calculate regression metrics (MSE, MAE, R²)",
        "category": "ML-Evaluation"
    },
    
    "org.knime.base.node.mine.scorer.entrop.EntropyNodeFactory": {
        "imports": [
            "import pandas as pd",
            "from sklearn.metrics import confusion_matrix"
        ],
        "code": """cm = confusion_matrix({input_var}['{actual_column}'], {input_var}['{predicted_column}'])
{output_var} = pd.DataFrame(cm)""",
        "description": "Generate confusion matrix",
        "category": "ML-Evaluation"
    },
    
    "org.knime.base.node.viz.roc.ROCCurveNodeFactory": {
        "imports": [
            "import pandas as pd",
            "from sklearn.metrics import roc_curve, auc"
        ],
        "code": """fpr, tpr, thresholds = roc_curve({input_var}['{actual_column}'], {input_var}['{probability_column}'])
roc_auc = auc(fpr, tpr)
{output_var} = pd.DataFrame({{'fpr': fpr, 'tpr': tpr, 'threshold': thresholds}})
print(f"AUC: {{roc_auc:.4f}}")""",
        "description": "Calculate ROC curve and AUC",
        "category": "ML-Evaluation"
    },
    
    # ==================== Data Preprocessing for ML ====================
    
    "org.knime.base.node.preproc.pmml.missingval.MissingValueNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """{output_var} = {input_var}.fillna({input_var}.mean())  # Numeric mean imputation
# For categorical: {input_var}.fillna({input_var}.mode().iloc[0])""",
        "description": "Handle missing values",
        "category": "ML-Preprocessing"
    },
    
    "org.knime.base.node.preproc.pmml.stdencode.PMMLStandardizeEncoderNodeFactory": {
        "imports": [
            "import pandas as pd",
            "from sklearn.preprocessing import StandardScaler"
        ],
        "code": """scaler = StandardScaler()
{output_var} = {input_var}.copy()
{output_var}[{columns}] = scaler.fit_transform({input_var}[{columns}])""",
        "description": "Standardize (z-score) numeric columns",
        "category": "ML-Preprocessing"
    },
    
    "org.knime.base.node.preproc.pmml.minmax.MinMaxNormalizerNodeFactory": {
        "imports": [
            "import pandas as pd",
            "from sklearn.preprocessing import MinMaxScaler"
        ],
        "code": """scaler = MinMaxScaler()
{output_var} = {input_var}.copy()
{output_var}[{columns}] = scaler.fit_transform({input_var}[{columns}])""",
        "description": "Min-Max normalize numeric columns",
        "category": "ML-Preprocessing"
    },
    
    "org.knime.base.node.preproc.pmml.categtoint.CategoryToIntegerNodeFactory": {
        "imports": [
            "import pandas as pd",
            "from sklearn.preprocessing import LabelEncoder"
        ],
        "code": """le = LabelEncoder()
{output_var} = {input_var}.copy()
for col in {columns}:
    {output_var}[col] = le.fit_transform({input_var}[col].astype(str))""",
        "description": "Encode categorical columns to integers",
        "category": "ML-Preprocessing"
    },
    
    "org.knime.base.node.preproc.pmml.onehottable.OneHotTableEncoderNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """{output_var} = pd.get_dummies({input_var}, columns={columns}, prefix={columns})""",
        "description": "One-Hot encode categorical columns",
        "category": "ML-Preprocessing"
    },
    
    # ==================== Train/Test Split ====================
    
    "org.knime.base.node.preproc.partition.PartitionNodeFactory": {
        "imports": [
            "import pandas as pd",
            "from sklearn.model_selection import train_test_split"
        ],
        "code": """train_df, test_df = train_test_split({input_var}, test_size={test_ratio}, random_state=42)""",
        "description": "Split data into train/test partitions",
        "category": "ML-Preprocessing"
    },
    
    "org.knime.base.node.preproc.crossvalidation.CrossValidationLoopStartNodeFactory": {
        "imports": [
            "import pandas as pd",
            "from sklearn.model_selection import KFold"
        ],
        "code": """kfold = KFold(n_splits={n_folds}, shuffle=True, random_state=42)
for fold_idx, (train_idx, val_idx) in enumerate(kfold.split({input_var})):
    train_fold = {input_var}.iloc[train_idx]
    val_fold = {input_var}.iloc[val_idx]
    # Process fold...""",
        "description": "K-Fold cross-validation loop start",
        "category": "ML-Validation"
    },
}


def get_ml_template(factory_class: str) -> Dict[str, Any] | None:
    """Get ML template by factory class."""
    return ML_TEMPLATES.get(factory_class)


def get_all_ml_factories() -> list:
    """Get list of all supported ML factory classes."""
    return list(ML_TEMPLATES.keys())
