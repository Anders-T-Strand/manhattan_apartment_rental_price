# Manhattan Apartment Rental Price Prediction

Predicting the monthly rent of Manhattan apartments from listing features (size, bedrooms, bathrooms, location, building age, and amenities), and comparing six regression algorithms on the same train/validation/test split to see which generalizes best.

## Data

`manhattan.csv` holds 3,539 StreetEasy listings across 32 Manhattan neighborhoods with 18 columns: `rent` (the target), six numeric features (`bedrooms`, `bathrooms`, `size_sqft`, `floor`, `building_age_yrs`, `min_to_subway`), eight binary amenity flags (`no_fee`, `has_roofdeck`, `has_washer_dryer`, `has_doorman`, `has_elevator`, `has_dishwasher`, `has_patio`, `has_gym`), plus `neighborhood` and `borough`. The raw file has no missing values.

Cleaning is light: drop the listings tagged as Long Island City rather than Manhattan, drop any listing with zero bathrooms, and treat a recorded floor of 0 as missing so it can be imputed later. New construction (building age 0) is kept. That removes 13 rows (0.4%), leaving 3,526.

## Approach

Rent is right-skewed (skewness 1.98), so the model predicts `log1p(rent)`, which drops the skew to 0.65 and keeps a few luxury listings from dominating the loss. Predictions are converted back to dollars for reporting.

The model uses 15 features: the six numeric columns, the eight amenity flags (already binary, so no encoding needed), and neighborhood. Neighborhood is target-encoded, where each area maps to the mean log-rent of the training listings in it. The encoding map is fit on the training split only, and unseen neighborhoods fall back to the global training mean, so there's no leakage from held-out data.

The split is 70/15/15 (2,469 / 528 / 529 rows), stratified on bedroom count (0, 1, 2, 3+) so each split has a similar unit mix. A random split is used since the data is a cross-sectional snapshot with no listing dates. The single missing floor value is filled with the training median, and features are standardized for the models that need it (Linear Regression, SVR, and the neural net); the tree-based models use unscaled, imputed data.

A correlation check lines up with intuition: `size_sqft` (r = 0.84) is the strongest predictor of log rent, followed by `bathrooms` (0.76) and `bedrooms` (0.67). Floor is mildly positive (0.26), building age mildly negative (−0.20), and everything else, including subway distance and the amenity flags, sits below 0.10 on its own. Neighborhood looks weak as a single correlation but spans a threefold range in median rent, from about $2,400 in Washington Heights to $7,722 in Soho, so it carries real signal once encoded. Among amenities, the largest median premiums go to patio (+$476), dishwasher (+$445), and in-unit washer/dryer (+$330); the no-fee flag is the most common (40% of listings) but shows no premium.

## Models

Six algorithms, eight configurations in total, all evaluated the same way:

- Linear Regression: OLS and Ridge (α=1), which came out nearly identical, so multicollinearity isn't a concern
- Decision Tree: a shallow tree (depth 6) and a deep one (depth 15)
- Random Forest: 200 trees, depth 20, 50% feature subsampling per split
- SVR: RBF kernel, C=10, ε=0.05, trained on a 2,000-row sample (~81% of training) because of quadratic memory scaling
- XGBoost: up to 1,000 rounds with early stopping, converged at 270 trees
- Neural Network: scikit-learn MLP, 128–64–32, ReLU, Adam, early stopping (halted at 129 of 300 epochs)

Metrics: RMSE and MAE in dollars per month (computed after converting predictions back from log space), and R² measured in log-price space. Raw-dollar R² collapses toward zero even for good models because listings above $15,000/month inflate the total sum of squares, so log-space R² better reflects performance across the full rent range.

## Results

Test-set performance, best first:

| Model | RMSE | MAE | R² (log) |
|---|---|---|---|
| XGBoost | $1,188 | $652 | 0.907 |
| Random Forest | $1,215 | $696 | 0.899 |
| Decision Tree (depth 15) | $1,224 | $754 | 0.869 |
| SVR (RBF) | $1,418 | $787 | 0.857 |
| Decision Tree (depth 6) | $1,366 | $801 | 0.854 |
| Ridge Regression | $1,960 | $888 | 0.844 |
| Linear Regression (OLS) | $1,962 | $888 | 0.844 |
| Neural Network | $1,970 | $933 | 0.831 |

XGBoost won across all three metrics, with a test R² of 0.907 and a MAE around $652/month (roughly 16% of the $4,000 median rent). Random Forest followed closely at R² 0.899. The two ensembles form a clear top tier; Linear Regression, SVR, and the neural net cluster between 0.831 and 0.857. The MLP placed last despite having the most parameters, which is the expected outcome on a structured dataset this small (3,526 rows, 15 features), where gradient-boosted trees have a built-in efficiency edge.

Both ensembles agree on what drives the prediction, with a telling rank swap. Random Forest (split-count importance) ranks size first at 0.524, then bathrooms 0.199 and neighborhood 0.097. XGBoost (gain-based importance) flips the top two, putting bathrooms at 0.575 ahead of size at 0.248, because bathroom count's discrete values make large, high-gain early splits. Either way, size and bathrooms are the top two and neighborhood is third. Subway proximity contributes almost nothing, which fits Manhattan's uniformly dense transit. Building age has a U-shaped effect (both the newest and oldest buildings rent above average) that the tree models capture but linear regression mostly misses, which is why its linear correlation is only −0.20 while its tree importance is meaningful.

## Files

| File | Description |
|---|---|
| `Strand_Anders_Project.ipynb` | The full pipeline: cleaning, EDA, feature engineering, six models, comparison |
| `manhattan.csv` | The dataset (3,539 StreetEasy Manhattan listings) |
| `requirements.txt` | Pinned package versions for the environment |

The notebook also writes its plots to disk as it runs: rent distribution, rent vs size and floor, median rent by neighborhood, amenity premiums, subway and building-age effects, the feature-correlation chart, the model comparison, and the feature-importance comparison.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook Strand_Anders_Project.ipynb
```

Run the cells top to bottom. The core dependencies are pandas, numpy, scikit-learn, xgboost, and matplotlib. Everything is seeded with `random_state=42`, so the numbers above should reproduce. Random Forest takes about a minute; the rest are quick.

## Notes

The dataset is a single cross-sectional snapshot with no listing dates, so seasonal or year-over-year price effects can't be modeled. The errors (MAE around 16–17% of median rent for the two ensembles) are good enough to flag mispriced listings or give a ballpark reference, but not precise enough for formal valuation.
