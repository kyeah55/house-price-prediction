import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb 

def load_data():
    data = pd.read_csv("Fiyat tahmin etme/data/train.csv")
    data = data[(data["SalePrice"] < 500000) & (data["SalePrice"] >= 50000)]
    return data

def preprocess_data(data, cols_to_drop):
    data = data.dropna(axis=0, subset=["SalePrice"])
    X = data.drop(["SalePrice"], axis=1).drop(cols_to_drop, axis=1, errors="ignore")
    y = data["SalePrice"]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

    cols_with_num = [col for col in X_train.columns if X_train[col].dtype in ["int32", "float32", "int64", "float64"]]
    cols_with_cat = [col for col in X_train.columns if X_train[col].dtype == "object"]

    return X_train, X_valid, y_train, y_valid, cols_with_num, cols_with_cat

def build_pipeline(model, cols_with_num, cols_with_cat):
    num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, cols_with_num),
        ("cat", cat_pipeline, cols_with_cat)
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline

def tune_gradient_boosting(X_train, y_train, preprocessor):
    from scipy.stats import randint, uniform

    param_dist = {
        "model__n_estimators": randint(50, 300),
        "model__max_depth": randint(3, 10),
        "model__learning_rate": uniform(0.01, 0.3),
        "model__min_samples_split": randint(2, 20),
        "model__min_samples_leaf": randint(1, 20)
    }

    base_model = GradientBoostingRegressor(random_state=0)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", base_model)
    ])

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='neg_mean_absolute_error',
        verbose=2,
        n_jobs=-1,
        random_state=0
    )

    search.fit(X_train, y_train)

    print("Best parameters found:", search.best_params_)
    print(f"Best MAE: {-search.best_score_:.2f}")

    return search.best_estimator_



if __name__ == "__main__":
    cols_to_drop = ['3SsnPorch', 'YrSold', 'PoolArea', 'EnclosedPorch', 'MiscVal',
                    'BsmtFinSF2', 'BedroomAbvGr', 'BsmtHalfBath', 'BsmtFullBath',
                    'MSSubClass', 'MoSold', 'Id']

    data = load_data()
    X_train, X_valid, y_train, y_valid, cols_with_num, cols_with_cat = preprocess_data(data, cols_to_drop)

    models = {
        "Random Forest": RandomForestRegressor(random_state=0),
        "Decision Tree": DecisionTreeRegressor(random_state=0),
        "Gradient Boosting": GradientBoostingRegressor(random_state=0),
        "Linear Regression": LinearRegression(),
        "XGBoost": xgb.XGBRegressor(random_state=0)
    }

    for name, model in models.items():
        pipeline = build_pipeline(model, cols_with_num, cols_with_cat)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_valid)
        mae = mean_absolute_error(y_valid, preds)
        print(f"{name} MAE: {mae:.2f}")


        # GradientBoosting için optimize edilmiş model denemesi
    print("\n🔍 Tuning Gradient Boosting...\n")
    preprocessor_only = build_pipeline(GradientBoostingRegressor(), cols_with_num, cols_with_cat).named_steps["preprocessor"]
    best_gb_model = tune_gradient_boosting(X_train, y_train, preprocessor_only)

    preds = best_gb_model.predict(X_valid)
    mae = mean_absolute_error(y_valid, preds)
    print(f"Tuned Gradient Boosting MAE: {mae:.2f}")
