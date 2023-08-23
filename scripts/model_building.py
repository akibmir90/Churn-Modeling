from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Train models
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)

logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, y_train)

xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)