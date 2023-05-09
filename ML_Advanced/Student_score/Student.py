import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

data = pd.read_csv("StudentScore.xls")
target = "math score"
corre = data.corr()
# sns.histplot(data[target])
# plt.title("Distribution")
# plt.show()

x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=100)
# print(data['race/ethnicity'].unique())

#Preprocessing

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

education_levels = ['some high school' , 'high school' , 'some college', "associate's degree","bachelor's degree","master's degree"]
gender = data['gender'].unique()
lunch = data['lunch'].unique()
test = data['test preparation course'].unique()

ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[education_levels,gender,lunch,test])) #thêm vào sau education_levels, ....
])

nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder())
])

# result = num_transformer.fit_transform(x_train[["reading score", "writing score"]])
# result = ord_transformer.fit_transform(x_train[["parental level of education"]])
# result = nom_transformer.fit_transform(x_train[["race/ethnicity"]])

preprocesssor = ColumnTransformer(transformers=[
    ("num_feature", num_transformer,['reading score','writing score']),
    ('ord_feature', ord_transformer, ['parental level of education', 'gender', 'lunch', 'test preparation course']),
    ('nom_feature', nom_transformer, ['race/ethnicity'])
])

reg = Pipeline(steps=[
    ('preprocessing_data', preprocesssor),
    ('regressor', LinearRegression())
])

reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)
# for i, j in zip(y_test, y_predict):
#     print(i, j)
print(r2_score(y_test, y_predict))
print(mean_absolute_error(y_test, y_predict))
print(mean_squared_error(y_test, y_predict))





