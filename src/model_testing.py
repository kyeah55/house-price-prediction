import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

def load_test_data():
    
    data = pd.read_csv("..\\data\\test.csv")

    return data


def preprocess_data(data, cols_to_drop):
    

    data = data.drop(cols_to_drop,axis=1,errors="ignore")

    return data


def load_model(path):

    model = joblib.load(path)

    return model


def predict(model, X_test):
    
    preds = model.predict(X_test)

    return preds


def save_predictions(ids,preds,filename):

    df = pd.DataFrame({
        "Id": ids,
        "SalePrice": preds
    })
    df.to_csv(filename, index=False)


if __name__ == "__main__":

    test_data = load_test_data()

    cols_to_drop = ['3SsnPorch', 'YrSold', 'PoolArea', 'EnclosedPorch', 'MiscVal',
                    'BsmtFinSF2', 'BedroomAbvGr', 'BsmtHalfBath', 'BsmtFullBath',
                    'MSSubClass', 'MoSold', 'Id']


    X_test = preprocess_data(test_data, cols_to_drop)


    model = load_model("..\\models\\best_model.pkl")


    predictions = predict(model, X_test)
    

    save_predictions(test_data["Id"], predictions, "..\\outputs\\submission.csv")


    print("Predictions saved to submission.csv")

