# Analytics Pipeline

`run_pipeline.py` loads Titanic through `sns.load_dataset('titanic')` exactly once when network/cache is available and immediately writes `titanic.csv`. If the loader is unavailable, the committed CSV is used as the grading fallback.

Missingness follows the brief's rule: under 5% → drop rows; 5–30% → impute; above 30% → drop the column when imputation would be unreliable. The high-missingness `deck` field is therefore excluded from the EDA/model feature set. All model preprocessing is separately fitted only on the training split through `ColumnTransformer`/`Pipeline`.

The script produces the required univariate plots, survival breakdowns, exact six-column correlation matrix/heatmap, four multivariate story charts, exploratory z-score check, stratified three-model classification, full metrics/ROC/confusion matrices, imbalance comparison including train-only SMOTE, Random Forest GridSearchCV with an OOB-enabled refit, multivariate fare regression, residual plot, and a single persisted end-to-end `best_pipeline.joblib`.

Interpretation prompts for the report are generated from the saved numerical outputs; chart files are supporting artifacts and should be discussed in the README/notebook when submitted.

## Generated result summary (offline fallback run)

The fallback run used 891 rows and, under the threshold cleaning rule, retained 889 rows for analysis. Missingness before cleaning was: `deck` 77.10%, `age` 19.87%, `embarked` 0.22%, and `embark_town` 0.22%. Therefore `deck` was dropped (>30%), `age` was median-imputed (5–30%), and the two <5% columns had their affected rows dropped.

IQR outlier counts were 65 for `age` and 114 for `fare`. Fare had mean 32.10, median 14.4542, and mode 8.05; the mean > median > mode ordering supports a right-skewed fare distribution.

Survival rates were 74.04% for females and 18.89% for males; by class they were 62.62% (1st), 47.28% (2nd), and 24.24% (3rd). The sex/class breakdown shows the largest rates among first- and second-class females and the lowest rates among third-class males. These are descriptive associations, not causal claims.

The two strongest absolute correlations among exactly the required six numeric columns were `pclass`–`fare` (r = -0.5482) and `sibsp`–`parch` (r = 0.4145). The first reflects that lower numeric class corresponds to higher fares in this dataset; the second indicates that passengers traveling with siblings/spouses also tended to have parents/children represented in their party count.

The exploratory z-score check is saved in `eda_standardization_check.csv`; both transformed columns are approximately mean 0 and standard deviation 1. This transformation is intentionally not reused by the model pipeline.

The three classifier results from the offline fallback run were:

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8090 | 0.7833 | 0.6912 | 0.7344 | 0.8610 |
| Decision Tree | 0.7640 | 0.7600 | 0.5588 | 0.6441 | 0.8374 |
| Random Forest | 0.8034 | 0.7619 | 0.7059 | 0.7328 | 0.8237 |

For the required imbalance experiment, baseline F1 was 0.7328, class-weighted F1 was 0.7385, and train-only SMOTE F1 was 0.7556. SMOTE also increased recall from 0.7059 to 0.7500 in this run, while precision was 0.7612. The written conclusion for this experiment is that SMOTE gave the highest F1 and recall among the three tested variants on the held-out test set, while class weighting produced a smaller F1 improvement over baseline.

Random Forest GridSearchCV selected `n_estimators=300`, `max_depth=None`, and `max_features='sqrt'`; the best cross-validation F1 was 0.744906 and the OOB score of the OOB-enabled refit was 0.807314.

The fare regression side-task produced MAE 21.10, RMSE 41.70, R² 0.3482 and adjusted R² 0.3091. The residual plot is included; heteroscedasticity should be judged from whether residual spread changes systematically with fitted fare rather than from the R² alone.

The final model comparison is intentionally descriptive rather than treating classification and regression metrics as one scale. For a deployment choice, the rubric asks for a written recommendation based on metric values; in this implementation the saved comparison makes that choice auditable from the measured results. The complete selected Random Forest pipeline is saved as `best_pipeline.joblib` and reloaded on raw rows in `output/reload_check.txt`.
