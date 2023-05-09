import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer

data = pd.read_csv('stroke_classification.csv')
y = data['stroke']
x = data.drop('stroke', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=100)

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

gender_value = ['Female', 'Male']

nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ('encoder', OneHotEncoder(sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num_feat", num_transformer, ['pat_id', 'age', 'hypertension', 'heart_disease', 'work_related_stress', 'urban_residence','avg_glucose_level','bmi','smokes']),
    ('ord_feat', nom_transformer, ['gender'])
])

clf = Pipeline(steps=[
    ("preprocessing_data", preprocessor),
    ("regressor", SVC())
])

clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
print(classification_report(y_test, y_predict))