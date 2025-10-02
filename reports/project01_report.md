# COE 379L Project 01 Report — Austin Animal Center Outcomes

## 1. Data Preparation Workflow

I began by loading the raw Austin Animal Center dataset (`project1.csv`) into pandas and auditing its structure. The raw file contains 131,165 records across the twelve attributes specified in the project brief, including identifiers, timestamps, categorical descriptors, and the target outcome label. After checking the dataframe shape, size, and schema, I documented the need to convert date-like strings (`Date of Birth`, `DateTime`, `MonthYear`) into `datetime` objects and to normalize the mixed-format `Age upon Outcome` field.

- Duplicate detection revealed repeated entries (17 rows). I removed duplicates while preserving the first occurrence to avoid double-counting outcomes. Missing-value analysis showed nulls primarily in `Outcome Type`, `Outcome Subtype`, and `Name`. I filled `Outcome Type` and `Outcome Subtype` with each column’s mode, and unnamed animals were labelled "Unknown". This keeps the dataset internally consistent without inventing unsupported values.

`Age upon Outcome` required special handling because values were stored as strings like `"2 years"` or `"3 weeks"`. I tokenized each entry, mapped the unit to a multiplier (days, weeks, months, years converted to days), and produced a numeric `AgeDays` feature. Any unresolved or missing ages were imputed with the median to maintain comparability across animals.

After parsing the timestamp fields, I engineered calendar-based features (`OutcomeYear`, `OutcomeMonth`, `BirthYear`, `BirthMonth`) to capture potential seasonal patterns. With these numeric descriptors in place, I dropped the original `Date of Birth`, `DateTime`, `MonthYear`, and string-based age columns to avoid redundant or non-numeric inputs. Per the project instructions—and to control feature explosion—I removed high-cardinality identifiers (`Breed`, `Name`) before one-hot encoding. Finally, I converted key categorical predictors (`Outcome Subtype`, `Animal Type`, `Sex upon Outcome`, `Color`) to the pandas `category` type and generated dummy variables using `pd.get_dummies(..., drop_first=True)`. The processed feature matrix was saved as `data/processed/clean_features.csv` for downstream modeling.

## 2. Exploratory Data Analysis Insights

- **Outcome balance.** Adoption and Transfer outcomes are both common, with Adoptions more frequent than Transfers. I maintained stratification when splitting the data to preserve class proportions.
- **Species mix.** Dogs dominate the records, followed by cats. Birds and other species appear rarely, indicating that species-specific features may mostly separate dogs and cats.
- **Sterilization status.** Most outcomes involve spayed or neutered animals. Intact animals are a minority, which may influence how strongly `Sex upon Outcome` contributes to predictions.
- **Age distribution.** The `AgeDays` histogram is sharply right-skewed: the bulk of animals are younger than roughly two years, but there is a long tail of senior animals. This suggests median-based imputation is appropriate and that age-related effects are nonlinear.
- **Temporal trends.** Aggregating outcomes by month highlights peaks in mid-2016 and a resurgence in 2024–2025. These bursts could reflect operational changes (e.g., adoption drives). Capturing `OutcomeMonth` and `OutcomeYear` numerically lets the models exploit any seasonality.

## 3. Modeling Procedure

- **Train/test split.** Using the engineered dataset, I separated features and target (`Outcome Type`) and performed an 80/20 stratified train/test split with `random_state=42` to guarantee reproducibility and preserve the Adoption/Transfer balance.
- **Pipelines.** Each model used a `Pipeline` with `StandardScaler(with_mean=False)` so the one-hot encoded matrix remained sparse while still scaling numeric inputs.
- **Baseline KNN.** Fit a K-Nearest Neighbors classifier with `k=5` as a simple baseline.
- **KNN with tuning (5-fold CV).** To satisfy the grid-search requirement without incurring excessive runtime, I drew a stratified sample of 2,000 training rows and performed a 5-fold `GridSearchCV` tuning only the `k` hyperparameter with `k ∈ {3, 5, 11}`. Scoring used F1 for the Adoption class. The best `k` was then refit on the full training set and evaluated on the held-out test set.
- **Linear model.** Trained a logistic regression classifier (`solver='lbfgs'`, `max_iter=200`) on the scaled features to provide a linear baseline. The solver emitted a convergence warning at 200 iterations; increasing `max_iter` would resolve it, but the model already performed well.

## 4. Model Performance Summary

| Model | Accuracy | Precision (Adoption) | Recall (Adoption) | F1 (Adoption) |
|-------|----------|----------------------|-------------------|---------------|
| KNN (k=5) | ≈0.86 | ≈0.87 | ≈0.93 | ≈0.90 |
| KNN (GridSearch, k∈{3,5,11}, 5-fold) | ≈0.87 | ≈0.86–0.88 | ≈0.94–0.95 | ≈0.90–0.91 |
| Logistic Regression | ≈0.88 | ≈0.87 | ≈0.96 | ≈0.91 |

*Notes:* Metrics are computed on the 26,230-record test set with Adoption treated as the positive class. The tuned KNN (5-fold CV over `k`) yielded modest gains over the baseline but did not surpass the linear model’s recall. Logistic regression achieved the strongest F1 and overall accuracy; although the solver warned about convergence at 200 iterations, performance was already strong and can be further stabilized by raising `max_iter`.

## 5. Priority Metric Justification

For an animal shelter, recall on the Adoption class is particularly important. A false negative (predicting Transfer when the outcome will be an Adoption) risks underestimating adoption demand and potentially delaying outreach or resource allocation that supports adoptions. Precision still matters—overstating adoption likelihood could misallocate efforts—but recall aligns more directly with the shelter’s goal of connecting animals to adopters. Consequently, I selected F1 (the harmonic mean of precision and recall) as the primary comparison metric and paid special attention to models with strong recall.*

## 6. Confidence and Limitations

- **Strengths.** The dataset is large and stratified; multiple algorithms were evaluated; cross-validation tuned the KNN hyperparameters; and feature engineering captured key temporal and demographic signals.
- **Limitations.** One-hot encoding creates a high-dimensional feature space, which can be sensitive to rare categories. Dropping `Breed` simplifies analysis (as required) but removes a potentially predictive attribute. The logistic regression assumes linear decision boundaries; non-linear models (e.g., gradient boosting) could uncover richer patterns if permitted. Additionally, model evaluation relies on a single temporal split; a time-based validation might better reflect deployment performance if the data distribution drifts.
- **Future work.** Incorporating more nuanced feature interactions, trying calibrated probability thresholds, or monitoring precision–recall trade-offs over time would improve deployment readiness. Regular retraining is advisable because outcome distributions may shift with shelter policy changes or seasonal intake variations.

Given the strong F1 score from logistic regression and the alignment between its high recall (≈0.96) and the shelter’s adoption goals, I am reasonably confident in the model’s ability to differentiate Adoption vs Transfer outcomes on similar data. Nonetheless, I recommend ongoing evaluation with fresh data and stakeholder feedback before relying on the model for high-stakes decisions.

