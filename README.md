# Salary Predictor with Explainable AI

Predict salaries for data and AI jobs in India (in LPA, lakhs per year) and **explain every prediction in plain words**, with SHAP, partial dependence plots, a prediction range and a what-if function.

Built in Google Colab. No API key needed.

> **The data is made up.** The 4,000 records follow a simple salary formula I wrote, so the scores below are flattering and the patterns show the method, not the real job market. The notebook also accepts your own `salary_data.csv` with the same columns.

## What it does

1. **Compares three models** (Linear Regression, Random Forest, Gradient Boosting) on a held-out test set.
2. **Explains the model with SHAP**, adding up the values of all one-hot columns that come from the same original feature. That turns confusing output like "education_Bachelors = False adds 1.07 LPA" into "education = Masters adds 0.91 LPA".
3. **Explains one person at a time**, with a waterfall chart and plain-language reasons.
4. **Shows partial dependence** (how salary changes with experience, skills and certifications).
5. **Gives a prediction range**, built from the model's cross-validated mistakes, so a prediction is never a single over-confident number.
6. **What-if predictor:** change a job title, city tier or number of certifications and see the new salary, the range and the top reasons. It warns when an input is outside what the model learned (for example 30 years of experience).
7. **Checks errors by group** (remote or not, city tier, education).

## Results from the run

| Model | MAE (LPA) | R2 |
|---|---|---|
| Gradient Boosting | 0.98 | 0.933 |
| Linear Regression | 1.13 | 0.915 |
| Random Forest | 1.23 | 0.901 |

**What drives salary (average SHAP impact, LPA):** job title 3.00, city tier 1.57, experience 1.49, education 1.26, company size 0.86, skills count 0.58, certifications 0.30, remote 0.16.

**Prediction range:** the model's typical mistakes run from -1.58 to +1.62 LPA, so every prediction gets a range about 3.2 LPA wide. On the test set, **82.2%** of real salaries landed inside it (target 80%).

**What-if examples:**

| Profile | Prediction (range) |
|---|---|
| Fresher Data Analyst, Bachelors, Tier 2 city, mid-size company, 6 skills, 2 certifications | 6.75 LPA (5.17 to 8.36) |
| Same person with 2 more certifications and 4 more skills | 8.96 LPA (7.38 to 10.58) |
| Same person in a Tier 1 city | 7.61 LPA (6.03 to 9.23) |
| Data Scientist, 3 years, Masters, Tier 1, large company | 21.30 LPA (19.72 to 22.91) |
| AI Engineer, 5 years, Masters, remote, Tier 1, large company | 26.51 LPA (24.93 to 28.12) |
| Analyst with 30 years of experience | 14.59 LPA, with a warning that this is outside the training range |

**Example explanation** (predicted 16.24 LPA, average 16.54): job title AI Engineer adds 3.81 LPA, Tier 3 city reduces 2.79, 2.0 years of experience reduces 1.87, a Masters degree adds 0.91.

**Error check by group:** the largest average error for any group was 0.34 LPA (Tier 3 cities). Remote workers were predicted 0.07 LPA too low on average and non-remote 0.20 too high. Small groups such as PhD (46 people) give noisy numbers.

## What the evaluation showed

- **A simple model is nearly as good.** Linear Regression reached R2 0.915 against 0.933 for Gradient Boosting, because the made-up salaries follow a clean formula. On real, noisier data the gap would likely be different, so this would need re-testing.
- **Grouped SHAP is what makes the explanations readable.** Per-column SHAP is correct but confusing for categories, and an earlier version explained a Masters graduate with "Bachelors adds LPA".
- **A range is more honest than one number.** Even for a fresher analyst, the realistic spread is about 3 LPA.

## Limitations

- **Made-up data.** Do not use these numbers as salary advice.
- The prediction range has the same width for everyone (3.2 LPA), so it is a rough guide. Real salaries are usually more spread out for senior roles.
- The error check is not a fairness audit: the data contains no protected attributes such as gender.
- SHAP shows what the model relies on, not what causes higher pay.

## Tech stack

scikit-learn, SHAP, pandas, matplotlib, seaborn.

## How to run

1. Open the notebook in Google Colab.
2. Run the cells from top to bottom. No API key is needed.
3. To use real data, upload `salary_data.csv` with the columns `job_title, experience_years, education, city_tier, company_size, skills_count, remote, certifications, salary_lpa`.

## Files the notebook creates

- `salary_shap_importance.csv`: importance by feature
- `salary_model_comparison.csv`: the three models compared
- `salary_test_predictions.csv`: predictions and errors on the test set

---

Built by **Akshat Kesharwani** | [GitHub](https://github.com/akshatkesharwani-info) | [Portfolio](https://akshatkesharwani-info.github.io/)
