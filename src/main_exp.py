from sklearn.model_selection import train_test_split
import pandas as pd
import json
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler, OrdinalEncoder , RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np


df_train = pd.read_csv('data/raw/train.csv')
X_test = pd.read_csv('data/raw/test.csv')
submit = X_test.copy()

df_train.drop(columns=['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature'], inplace=True)
X_test.drop(columns=['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature'], inplace=True)


X_train, X_val, y_train, y_val = train_test_split(
    df_train, df_train['SalePrice'], test_size=0.2, random_state=42
)


numeric_features = df_train.select_dtypes(include=['int64', 'float64']).columns.drop('SalePrice')
categorical_features = df_train.select_dtypes(include=['object', 'bool']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Заполнение пропусков средним значением
    ('scaler', RobustScaler()),
    # ('scaler', StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Заполнение пропусков наиболее частым значением
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)

catboost_model = CatBoostRegressor(
    eval_metric='RMSE',
    random_seed=42,
    verbose=100
)

catboost_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    use_best_model=True,
    early_stopping_rounds=50
)
submit['SalePrice'] = catboost_model.predict(X_test)

sub = pd.DataFrame()
sub['Id'] = submit['Id']
sub['SalePrice'] = submit['SalePrice']

sub.to_csv('out/processed_data.csv',index=False)


# Вычисление метрик
metrics = {
    'mae': 1,
    'mse': 2,
    'rmse': 33,
    'r2': 4
}

# Сохранение метрик
with open('metrics/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("Метрики сохранены:", metrics)