# Compare F1-scores and select the model with the highest F1-score
models = {
    'Random Forest': rf_f1,
    'Logistic Regression': lr_f1,
    'XGBoost': xgb_f1
}

best_model = max(models, key=models.get)
print("Best Model:", best_model)
