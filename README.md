# Titanic Survival Analysis

Exploratory Data Analysis, data cleaning, and feature engineering on the Titanic passenger dataset, preparing it for a Machine Learning pipeline.

## Project Structure

```
├── Titanic.csv                      # Raw dataset
├── Titanic_Survival_Analysis.ipynb  # Main notebook: cleaning, EDA, feature engineering
├── Titanic_processed.csv            # Final cleaned + feature-engineered dataset
├── Summary_Report.md                # Written summary of findings and decisions
└── README.md
```

## Setup / How to Run

1. Clone this repository:
   ```
   git clone <this-repo-url>
   cd titanic-survival-analysis
   ```
2. Install the required libraries:
   ```
   pip install pandas numpy matplotlib seaborn
   ```
3. Open `Titanic_Survival_Analysis.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
4. Run all cells top to bottom. `Titanic.csv` must be in the same folder as the notebook (already included in this repo).

The notebook is already executed with outputs and visualizations embedded, so it can also be viewed directly on GitHub without running it.

## What This Project Covers

### 1. Data Cleaning
- Missing `Age` values (~20%) imputed using the median age **within each passenger class**, since median age varies meaningfully by class.
- Missing `Embarked` values (2 rows) filled with the most frequent port.
- `Cabin` (~77% missing) converted into a binary `Has_Cabin` feature instead of being imputed or dropped outright.
- `Fare` outliers kept as genuine data and log-transformed (`Fare_log`) rather than removed.
- No duplicate rows found.

### 2. Exploratory Data Analysis
Includes visualizations and insights on survival rate by sex, passenger class, age, port of embarkation, and family size, plus a correlation heatmap of numeric features. Full findings are in `Summary_Report.md` and inline in the notebook.

### 3. Feature Engineering
| Feature | Built From | Purpose |
|---|---|---|
| `Has_Cabin` | Cabin | Signal from missingness pattern |
| `Title` | Name | Extracted title (Mr/Mrs/Miss/Master/Rare) |
| `FamilySize`, `IsAlone` | SibSp + Parch | Combined family-size signal |
| `AgeGroup` | Age | Binned age to capture threshold effects |
| `Fare_log` | Fare | Log-transformed to reduce skew |

All categorical variables are encoded (binary for `Sex`, one-hot for `Embarked`/`Title`/`AgeGroup`), and the final dataset (`Titanic_processed.csv`) is fully numeric with no missing values — ready for direct use in a machine learning model.

## Key Findings
- **Sex** was the strongest single predictor — female survival rate ~74% vs. male ~19%.
- **Passenger class** strongly predicted survival: 1st class ~63%, 2nd ~47%, 3rd ~24%.
- **Family size** showed a "sweet spot" — small-to-medium families survived better than solo travelers or very large families.

Full details and justification for every decision are in `Summary_Report.md`.

## Tools Used
Python, Jupyter Notebook, pandas, numpy, matplotlib, seaborn

**Note:** If the notebook doesn't preview inline on GitHub, view it via [nbviewer](https://nbviewer.org/github/abhishekshalot008-hub/titanic-survival-analysis/blob/main/Titanic_Survival_Analysis.ipynb) or download and open in Jupyter/Colab.

