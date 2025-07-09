import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
import joblib


def calculateMean(y_valid,preds):
    mae = mean_absolute_error(y_valid,preds)
    print(f"Mean Absolute Error: {mae:.2f}")

def featureElimiation(X_train,y_train,X_valid,y_valid,model_pipeline, cols_to_test):    
    import copy

    base_model = copy.deepcopy(model_pipeline)
    base_model.fit(X_train,y_train)
    base_preds = base_model.predict(X_valid)
    base_mae = mean_absolute_error(y_valid,base_preds)

    print(f"Base MAE: {base_mae:.2f}")

    cols_to_drop_final=[]

    for col in cols_to_test:
        print(f"Trying without {col}...")

        X_train_temp = X_train.drop(col,axis=1)
        X_valid_temp = X_valid.drop(col,axis=1)

        cols_with_number_temp = [c for c in X_train_temp.columns if X_train_temp[c].dtype in ["int32","float32","int64","float64"]]
        cols_with_cat_temp = [c for c in X_train_temp.columns if X_train_temp[c].dtype == "object"]

        num_pipeline_temp = Pipeline(steps=[
            ("imputer",SimpleImputer(strategy="median"))
        ])

        cat_pipeline_temp = Pipeline(steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("onehot",OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor_temp = ColumnTransformer(transformers=[
            ("num",num_pipeline_temp,cols_with_number_temp),
            ("cat",cat_pipeline_temp,cols_with_cat_temp)
        ])

        model_pipeline_temp = Pipeline(steps=[
            ("preprocessor",preprocessor_temp),
            ("model",RandomForestRegressor(random_state=0))
        ])


        model_pipeline_temp.fit(X_train_temp,y_train)
        preds_temp = model_pipeline_temp.predict(X_valid_temp)
        mae_temp = mean_absolute_error(y_valid,preds_temp)

        print(f"MAE without {col}: {mae_temp:.2f}")

        if mae_temp<base_mae:
            print(f"Removing {col} improved MAE from {base_mae:.2f} to {mae_temp:.2f}")
            cols_to_drop_final.append(col)
            base_mae = mae_temp
            X_train = X_train_temp
            X_valid = X_valid_temp
            model_pipeline = model_pipeline_temp
    
    return cols_to_drop_final, X_train, X_valid, model_pipeline

def featureElimiation_independent(X_train, y_train, X_valid, y_valid, model_pipeline, cols_to_test):
    import copy

    # copy 
    base_model = copy.deepcopy(model_pipeline)
    base_model.fit(X_train, y_train)
    base_preds = base_model.predict(X_valid)
    base_mae = mean_absolute_error(y_valid, base_preds)

    print(f"Base MAE: {base_mae:.2f}")

    cols_to_drop_final = []

    for col in cols_to_test:
        print(f"Trying without {col}...")

        # Each time drop the column from original data
        X_train_temp = X_train.drop(col, axis=1).copy()
        X_valid_temp = X_valid.drop(col, axis=1).copy()

        # Prepare pipeline according to column types
        cols_with_number_temp = [c for c in X_train_temp.columns if X_train_temp[c].dtype in ["int32", "float32", "int64", "float64"]]
        cols_with_cat_temp = [c for c in X_train_temp.columns if X_train_temp[c].dtype == "object"]

        num_pipeline_temp = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ])

        cat_pipeline_temp = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor_temp = ColumnTransformer(transformers=[
            ("num", num_pipeline_temp, cols_with_number_temp),
            ("cat", cat_pipeline_temp, cols_with_cat_temp)
        ])

        model_pipeline_temp = Pipeline(steps=[
            ("preprocessor", preprocessor_temp),
            ("model", RandomForestRegressor(random_state=0))
        ])

        model_pipeline_temp.fit(X_train_temp, y_train)
        preds_temp = model_pipeline_temp.predict(X_valid_temp)
        mae_temp = mean_absolute_error(y_valid, preds_temp)

        print(f"MAE without {col}: {mae_temp:.2f}")

        if mae_temp < base_mae:
            print(f"Removing {col} improved MAE from {base_mae:.2f} to {mae_temp:.2f}")
            cols_to_drop_final.append(col)

    return cols_to_drop_final


# setting data
trainData = pd.read_csv(r"C:\Users\srwde\Desktop\Projects\Fiyat tahmin etme\data\train.csv")
testData = pd.read_csv(r"C:\Users\srwde\Desktop\Projects\Fiyat tahmin etme\data\test.csv")

# only 9 data will be deleted which is nothing
trainData = trainData[(trainData["SalePrice"] < 500000) & (trainData["SalePrice"]>=50000)]

# setting x and y
filtered_train = trainData.dropna(axis=0,subset=["SalePrice"])
final_cols_to_drop = ['3SsnPorch', 'YrSold', 'PoolArea', 'EnclosedPorch', 'MiscVal', 'BsmtFinSF2', 'BedroomAbvGr', 'BsmtHalfBath', 'BsmtFullBath', 'MSSubClass', 'MoSold', 'Id']


X = filtered_train.drop("SalePrice",axis=1).drop(final_cols_to_drop,axis=1)
y = filtered_train["SalePrice"]

# split the data
X_train, X_valid, y_train, y_valid = train_test_split(X,y,train_size=0.8,random_state=0)


# useful columns 
cols_with_number = [col for col in X_train.columns if X_train[col].dtype in ["int32","float32","int64","float64"]]
cols_with_cat = [col for col in X_train.columns if X_train[col].dtype == "object"]

# pipelines
num_pipeline = Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="median"))
])

cat_pipeline = Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("onehot",OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num",num_pipeline,cols_with_number),
    ("cat",cat_pipeline,cols_with_cat)
])

model_pipeline=Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("model",RandomForestRegressor(random_state=0))
])


# Grid Search
param_grid = {
    "model__n_estimators": [50, 100, 200],          # Number of trees
    "model__max_depth": [None, 10, 20],             # Maximum depth
    "model__min_samples_split": [2, 5, 10],         # Minimum samples to split a node
    "model__min_samples_leaf": [1, 2, 4]            # Minimum samples at leaf node
}

grid_search = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=5,                # 5-fold cross validation
    scoring="neg_mean_absolute_error",  # Optimizing MAE
    n_jobs=-1            # Use all CPU cores
)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best score (MAE):", -grid_search.best_score_)

best_model = grid_search.best_estimator_ # already trained!


# fit predict and mean ,,, best model already trained so no need to fit again

preds = best_model.predict(X_valid)

calculateMean(y_valid,preds)

# Remove columns based on correlation and feature importance analysis

cols_to_drop = [
    "3SsnPorch", "YrSold", "PoolArea", "EnclosedPorch", "MiscVal", "BsmtFinSF2", 
    "BedroomAbvGr", "BsmtHalfBath", "BsmtFullBath", "MSSubClass", "HalfBath", 
    "ScreenPorch", "Fireplaces", "LowQualFinSF", "KitchenAbvGr",
    "MoSold", "Id", "OverallCond"
]

final_cols_to_drop = ['3SsnPorch', 'YrSold', 'PoolArea', 'EnclosedPorch', 'MiscVal', 'BsmtFinSF2', 'BedroomAbvGr', 'BsmtHalfBath', 'BsmtFullBath', 'MSSubClass', 'MoSold', 'Id']

joblib.dump(best_model, 'best_model.pkl')

'''
cols_removed, X_train_new, X_valid_new, model_pipeline_new = featureElimiation(
    X_train, y_train, X_valid, y_valid, model_pipeline, cols_to_drop
)

preds_new = model_pipeline_new.predict(X_valid_new)
calculateMean(y_valid, preds_new)
'''

"""
cols_removed_independent = featureElimiation_independent(
    X_train, y_train, X_valid, y_valid, model_pipeline, cols_to_drop
)

print("Columns suggested for removal in independent trials:", cols_removed_independent)
"""

# For better performance

#            *FEATURE IMPORTANCE*
#             we check columns importance and remove them one by one


"""
ohe = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'] # access one-hot encoded categorical columns
ohe_columns = list(ohe.get_feature_names_out(cols_with_cat)) # convert categorical feature names to list

all_columns = cols_with_number + ohe_columns

importances=pd.Series(model_pipeline.named_steps["model"].feature_importances_ , index=all_columns).sort_values(ascending=False) # access feature importances

'''
plt.figure(figsize=(10,6))
importances[-20:].plot(kind='barh') # horizontal bar chart
plt.gca().invert_yaxis()
plt.title("Feature Importances (Top 20)")
# plt.show()
'''

original_cols = X_train.columns
low_importances= importances[(importances<0.003) & (importances.index.isin(original_cols))].index.tolist()


results = []

for col in low_importances:
    X_train_temp = X_train.drop([col],axis=1)
    X_valid_temp = X_valid.drop([col],axis=1)

    cols_with_number_temp = [c for c in X_train_temp.columns if X_train_temp[c].dtype in ["int32","float32","int64","float64"]]
    cols_with_cat_temp = [c for c in X_train_temp.columns if X_train_temp[c].dtype == "object"]

    num_pipeline_temp = Pipeline(steps=[
        ("imputer",SimpleImputer(strategy="median"))
    ])

    cat_pipeline_temp = Pipeline(steps=[
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor_temp = ColumnTransformer(transformers=[
        ("num",num_pipeline_temp,cols_with_number_temp),
        ("cat",cat_pipeline_temp,cols_with_cat_temp)
    ])

    model_pipeline_temp = Pipeline(steps=[
        ("preprocessor",preprocessor_temp),
        ("model",RandomForestRegressor(random_state=0))
    ])

    model_pipeline_temp.fit(X_train_temp,y_train)
    preds_temp = model_pipeline_temp.predict(X_valid_temp)

    mae =   mean_absolute_error(y_valid,preds_temp)

    results.append((col,mae))

results_sorted = sorted(results, key=lambda x: x[1])

print("MAE values after removing low importance columns:\n")
for col, mae in results_sorted:
    print(f"{col}: MAE = {mae:.2f}")

"""

#            *CORRELATION ANALYSIS

'''
cols_with_number_korelasyon= cols_with_number.copy()

if "SalePrice" not in cols_with_number_korelasyon:
    cols_with_number_korelasyon.append("SalePrice")

numeric_df = trainData[cols_with_number_korelasyon]

corr_matrix = numeric_df.corr()
corr_with_target = corr_matrix["SalePrice"].sort_values(ascending=False)

print("Correlation with SalePrice:\n",corr_with_target)
'''
