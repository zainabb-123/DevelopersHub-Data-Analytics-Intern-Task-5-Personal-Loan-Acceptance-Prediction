# Bank Term Deposit Subscription Prediction

## Task Objective

The goal of this project is to predict whether a bank client will subscribe to a term deposit (`y` = yes/no) based on their demographic and campaign-related attributes (age, job, marital status, education, loan history, contact details, and prior campaign outcomes). This is a binary classification problem aimed at helping a bank's marketing team identify which clients are most likely to respond positively to a term deposit offer, so future campaigns can be targeted more efficiently.

## Approach

The dataset (`bank.csv`, 11,162 records, 17 columns) is the well-known Bank Marketing dataset, with the target column renamed from `deposit` to `y`. No missing values were present across any column. The workflow combines exploratory analysis with two classification models:

1. **Data loading and inspection** – The data was loaded and checked for structure and data types using `df.info()`, confirming a mix of 7 numerical columns (age, balance, day, duration, campaign, pdays, previous) and 10 categorical columns (job, marital, education, default, housing, loan, contact, month, poutcome, and the target).
2. **Exploratory data analysis (EDA)** – Several visualizations were produced to understand the data and its relationship to the target: the distribution of the target variable, the distribution of age and job categories, marital status breakdown, and crosstabs of age, job, and marital status against subscription outcome.
3. **Preprocessing** – Numerical features were standardized using `StandardScaler`, and categorical features were one-hot encoded using `OneHotEncoder`, combined in a single `ColumnTransformer`.
4. **Train/test split** – The data was split 70/30 into training and testing sets using `train_test_split` with `stratify=y` to preserve the class balance, and `random_state=42` for reproducibility.
5. **Modeling** – Two classifiers were trained, each inside its own `Pipeline` (preprocessing + classifier), so transformations are applied consistently:
   - **Logistic Regression** (`liblinear` solver)
   - **Decision Tree Classifier**
6. **Evaluation** – Each model was evaluated on the held-out test set (3,349 records) using a classification report (precision, recall, F1-score), a confusion matrix, and an ROC curve with AUC.

## Results and Insights

**Target balance:** The dataset is reasonably balanced, with about 47% of clients subscribing ("yes") and 53% not subscribing ("no"), which means accuracy is a meaningful metric here without heavy class-imbalance correction.

**Model performance:**

| Model | Accuracy | Precision (yes) | Recall (yes) | F1 (yes) | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 83% | 0.83 | 0.80 | 0.81 | 0.91 |
| Decision Tree | 79% | 0.79 | 0.76 | 0.77 | 0.79 |

Logistic Regression clearly outperformed the Decision Tree on every metric, including a notably stronger ROC AUC (0.91 vs. 0.79), meaning it separates subscribers from non-subscribers much more reliably. Its confusion matrix shows 1,503 true negatives and 1,263 true positives out of 3,349 test cases, with a relatively low and balanced error rate (259 false positives, 324 false negatives). The Decision Tree, while still reasonably accurate, made more errors in both directions (327 false positives, 378 false negatives) and its ROC curve is far more jagged/piecewise, consistent with a single tree's tendency to overfit and generalize less smoothly than a linear model with regularization.

**Key drivers of subscription, from the EDA:**
- **Job category matters a lot.** Students and retirees are the only job categories where the number of subscribers exceeds non-subscribers; every other job category (management, blue-collar, technician, admin, services, self-employed, unemployed, entrepreneur, housemaid) sees more "no" than "yes" responses. This suggests clients outside the regular workforce (with more flexible finances or different saving priorities) are disproportionately likely to subscribe.
- **Single clients subscribe at a higher rate than married or divorced clients.** Among single clients, "yes" responses outnumber "no," whereas married clients show the opposite pattern by a wide margin (roughly 2,750 yes vs. 3,600 no), and divorced clients are roughly even.
- **Age is a weak standalone signal.** The age distributions for subscribers and non-subscribers largely overlap (median age ~38 vs. ~39), though subscribers include more older outliers (up to ~95 years old), echoing the retiree pattern seen in the job breakdown.

**Takeaway:** Demographic and occupation-based segments (notably students, retirees, and single clients) are meaningfully more receptive to term deposit offers, and a regularized linear model (Logistic Regression) generalizes better than a single Decision Tree on this data. For a production use case, the Logistic Regression model would be the better choice as-is; further gains could likely come from an ensemble method like Random Forest or Gradient Boosting, which typically reduce a single tree's overfitting while retaining the ability to capture non-linear interactions (e.g., between job and marital status).
