Metrics Selection:

- MAE: Mean Absolute Error - Robust to outliers and doesn't penalize large errors as heavily as RMSE
- RMSE: Root Mean Square Error - penalizes larger errors more than MAE. Good when large mistake are worse than the small one
- R^2: Proportion of variance explained by the model - Measures how well your predictions explain the variance in the data

The model learns the continuous mapping, regression metrics should be used to assess the performance of model

Target Value:
- MAE: 2–5 days, Model is off by ~2–5 days on average
- R²: 0.85 – 0.95+, 85–95% of age variance explained

For the Loss Function:
`nn.SmoothL1Loss`:

Also known as Huber Loss, it is a hybrid loss that combines:
- MSE (Mean Squared Error) for small errors (for smooth convergence)
- MAE (Mean Absolute Error) for large errors (for robustness to outliers)

why this loss is suitable for the age task:
- Less sensitive to outliers - Real-world image data has noise or mislabels
- Stable gradients - Converges well even with varying learning rates
- Better than pure MSE - Doesn’t excessively penalize large but valid errors
- Continuous-valued target - Ideal for regression (age prediction = continuous target)


