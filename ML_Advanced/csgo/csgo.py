import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer

data = pd.read_csv("csgo.csv")
data = data.drop(['team_a_rounds', 'team_b_rounds','day','month','year','date', 'match_time_s', 'hs_percent'], axis=1)
# print(data)
x = data.drop('result', axis=1)
y = data['result']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=100)

#preprocessing_data
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy='median')),
    ("scaler", StandardScaler())
])

nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy='most_frequent')),
    ("encoder", OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ("num_feat", num_transformer, ['wait_time_s', 'ping', 'kills', 'assists', 'deaths', 'mvps', 'points']),
    ("nom_feat", nom_transformer, ['map'])
])

clf = Pipeline(steps=[
    ("preprocess_data", preprocessor),
    ("classifier", SVC())
])

clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
print(classification_report(y_test, y_predict))




