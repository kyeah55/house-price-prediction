import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import joblib

def load_data():
    
    data = pd.read_csv("..\\data\\train.csv")

    return data


def preprocess_data(data, cols_to_drop):
    
    data = data[(data["SalePrice"]<500000) & (data["SalePrice"]>=50000)]
    filtered_data = data.dropna(axis=0, subset=["SalePrice"])

    X = filtered_data.drop(["SalePrice"],axis=1).drop(cols_to_drop,axis=1,errors="ignore")
    y = filtered_data["SalePrice"]

    X_train, X_valid, y_train, y_valid = train_test_split(X,y,random_state=0)

    cols_with_num = [col for col in X_train.columns if X_train[col].dtype in ["int32","float32","int64","float64"]]
    cols_with_cat = [col for col in X_train.columns if X_train[col].dtype == "object"]

    return X_train, X_valid, y_train, y_valid, cols_with_num, cols_with_cat


def build_pipeline(cols_with_num, cols_with_cat):

    num_pipeline = Pipeline(steps=[
        ("imputer",SimpleImputer(strategy="mean"))
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num",num_pipeline, cols_with_num),
        ("cat",cat_pipeline,cols_with_cat)
    ])

    model_pipeline = Pipeline(steps=[
        ("preprocessor",preprocessor),
        ("model",GradientBoostingRegressor(random_state=0))
    ])
    
    return model_pipeline


def train_model(pipeline, X_train, y_train):
    
    pipeline.fit(X_train,y_train)

    return pipeline


def evaluate_model(pipeline, X_valid, y_valid):
    preds = pipeline.predict(X_valid)
    mae = mean_absolute_error(y_valid,preds)

    return mae


def save_model(pipeline, filename):
    joblib.dump(pipeline, filename)


if __name__ == "__main__":

    data = load_data()

    cols_to_drop = ['3SsnPorch', 'YrSold', 'PoolArea', 'EnclosedPorch', 'MiscVal',
                    'BsmtFinSF2', 'BedroomAbvGr', 'BsmtHalfBath', 'BsmtFullBath',
                    'MSSubClass', 'MoSold', 'Id']
    
    X_train, X_valid, y_train, y_valid, cols_with_num, cols_with_cat = preprocess_data(data, cols_to_drop)

    pipeline = build_pipeline(cols_with_num, cols_with_cat)

    trained_model = train_model(pipeline, X_train, y_train)

    mae = evaluate_model(trained_model, X_valid, y_valid)
    print(f"Validation MAE: {mae:.2f}")

    save_model(trained_model, "..\\models\\best_model.pkl")

