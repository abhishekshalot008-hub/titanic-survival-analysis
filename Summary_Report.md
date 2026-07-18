# Titanic Survival Analysis — Summary Report

## Overview
This project analyzes the Titanic passenger dataset (891 passengers, 12 original columns) to understand the factors associated with survival, and prepares a cleaned, feature-engineered dataset ready for machine learning.

## 1. Data Cleaning

| Issue | Column | Action Taken | Justification |
|---|---|---|---|
| Missing values (177, ~20%) | Age | Imputed using median age **within each passenger class** | Median age varies meaningfully by class (1st ≈ 37, 2nd ≈ 29, 3rd ≈ 24); a single global median would distort this pattern |
| Missing values (2) | Embarked | Filled with mode (most frequent port) | Only 2 missing rows — safe, low-impact fix |
| Missing values (687, ~77%) | Cabin | Dropped raw column, replaced with binary `Has_Cabin` feature | Too sparse to impute reliably, but *whether* a cabin was recorded still carries signal |
| Duplicates | — | None found | No action needed |
| Outliers | Fare | Kept, addressed via log transform (`Fare_log`) instead of removal | High fares are genuine data tied to class/survival, not errors |

## 2. Key EDA Insights

1. **Sex is the strongest single predictor of survival.** Female passengers survived at ~74%, compared to ~19% for male passengers.
2. **Passenger class strongly predicts survival.** Survival rate falls from ~63% (1st class) to ~47% (2nd class) to ~24% (3rd class).
3. **Children had a survival advantage.** Passengers under roughly age 10 show a higher survival share relative to their group size.
4. **Port of embarkation correlates with survival** (Cherbourg ~55%, Queenstown ~39%, Southampton ~34%) — most likely a proxy for the class composition of passengers boarding at each port, not a direct effect of the port itself.
5. **Family size shows a "sweet spot."** Solo travelers and very large families (6+) had lower survival rates than small-to-medium family groups (2-4) — motivating the engineered `FamilySize` and `IsAlone` features.

## 3. Feature Engineering

| New Feature | Derived From | Purpose |
|---|---|---|
| `Has_Cabin` | Cabin (missingness) | Captures signal from cabin data without needing the actual (mostly missing) cabin number |
| `Title` | Name | Extracts hidden age/sex/social-status signal from passenger names, simplified into Mr/Mrs/Miss/Master/Rare |
| `FamilySize`, `IsAlone` | SibSp + Parch | Combined signal was clearer in EDA than either raw column alone |
| `AgeGroup` | Age | Captures threshold effects (e.g. child survival advantage) more directly than continuous age |
| `Fare_log` | Fare | Log transform reduces the influence of Fare's right-skew/outliers |

**Encoding:** `Sex` was binary-encoded; `Embarked`, `Title`, and `AgeGroup` were one-hot encoded (`drop_first=True` to avoid redundant columns). `Name`, `Ticket`, `PassengerId`, and the raw `Fare` were dropped after their useful information was extracted into the features above.

## 4. Final Dataset
- **Shape:** 891 rows × 20 columns
- **Missing values:** none
- **All features:** fully numeric, ready for direct use with any standard ML model (Logistic Regression, Random Forest, etc.)
- **File:** `Titanic_processed.csv`

## 5. Conclusion
Sex, passenger class, and family size emerged as the clearest, most consistent predictors of survival in this dataset, closely matching well-documented historical accounts of the disaster (women/children prioritized, first-class passengers had better lifeboat access). The cleaned and engineered dataset preserves these real-world patterns while being fully prepared for a machine learning pipeline.
