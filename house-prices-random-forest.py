from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import norm
train_data = pd.read_csv('data/train.csv')

y_train = train_data['SalePrice']

X_test = pd.read_csv('data/test.csv')
test_id = X_test['Id']
X_test = X_test.drop(['Id'], axis=1)

# data = pd.concat([train_data, test_data], axis=0, sort=False)
data = train_data.drop(['Id', 'SalePrice'], axis=1)

categorical_features = [col for col in data.columns if data[col].dtypes == 'O']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_features = [col for col in data.columns if data[col].dtypes != 'O']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('categorical', categorical_transformer, categorical_features),
        ('numerical', numerical_transformer, numerical_features)
    ])

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor())])

# X_train = data[:len(y_train)]
# X_test = data[:len(y_train)]

model.fit(data, y_train)

y_preds = model.predict(X_test)


sub = pd.DataFrame()
sub['Id'] = test_id
sub['SalePrice'] = y_preds
sub.to_csv('housesub.csv',index=False)
