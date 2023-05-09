from tabulate import tabulate
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Read from df and visualize
df = pd.read_csv(filepath_or_buffer="data/train.csv", header=0)
df = df[['PassengerId', 'Survived', 'Pclass', 'Sex','Age', 'SibSp', 'Parch', 'Fare']]

df.dropna(inplace=True)
df = df.reset_index(drop=True)

#Tạo cột mới tính tổng người thân
'''
df['No_Relatives'] = df[['SibSp','Parch']].sum(axis=1)

df['No_Relatives'] = df['SibSp'] + df['Parch']
'''
df['No_Relatives'] = df[['SibSp','Parch']].sum(axis=1)
df.drop(['SibSp','Parch'], axis=1, inplace=True)

#Đổi Male = 1, Female = 0
'''
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)

gender_dict = {'male': 0, 'female': 1}
df['Sex'] = df['Sex'].map(gender_dict)

for idx, row in df.iterrows():
    if row['Sex'] == 'male':
        df.loc[idx, 'Sex'] = 1
    else:
        df.loc[idx, 'Sex'] = 0
'''
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

# print(tabulate(df, headers='keys', tablefmt='psql'))

#Task1
'''
new_df = df.groupby('Pclass')['Age'].mean()
print(new_df)
'''

#Task2
'''
new_df = df[(df['Sex']==0) & (df['Pclass']==1)]['PassengerId'].count()
print(new_df)
'''

#Task3
'''
range_0_10 = df[(df['Age']>0) & (df['Age']<=10)]['PassengerId'].count()
range_10_20 = df[(df['Age']>10) & (df['Age']<=20)]['PassengerId'].count()
range_20_30 = df[(df['Age']>20) & (df['Age']<=30)]['PassengerId'].count()
range_30_40 = df[(df['Age']>30) & (df['Age']<=40)]['PassengerId'].count()
range_40_50 = df[(df['Age']>40) & (df['Age']<=50)]['PassengerId'].count()
range_50 = df[df['Age']>50]['PassengerId'].count()

range_0_10_Death = df[(df['Age']>0) & (df['Age']<=10) & (df['Survived']==0)]['PassengerId'].count()
range_10_20_Death = df[(df['Age']>10) & (df['Age']<=20) & (df['Survived']==0)]['PassengerId'].count()
range_20_30_Death = df[(df['Age']>20) & (df['Age']<=30) & (df['Survived']==0)]['PassengerId'].count()
range_30_40_Death = df[(df['Age']>30) & (df['Age']<=40) & (df['Survived']==0)]['PassengerId'].count()
range_40_50_Death = df[(df['Age']>40) & (df['Age']<=50) & (df['Survived']==0)]['PassengerId'].count()
range_50_Death = df[(df['Age']>50) & (df['Survived']==0)]['PassengerId'].count()

death_rate = [
    range_0_10_Death/range_0_10, 
    range_10_20_Death/range_10_20, 
    range_20_30_Death/range_20_30, 
    range_30_40_Death/range_30_40, 
    range_40_50_Death/range_40_50, 
    range_50_Death/range_50_Death 
]

print(*death_rate)
'''

#Task4
'''
range_1 = df[(df['Pclass']==1)]['PassengerId'].count()
range_2 = df[(df['Pclass']==2)]['PassengerId'].count()
range_3 = df[(df['Pclass']==3)]['PassengerId'].count()

range_1_survived = df[(df['Pclass']==1) & (df['Survived']==1)]['PassengerId'].count()
range_2_survived = df[(df['Pclass']==2 & (df['Survived']==1))]['PassengerId'].count()
range_3_survived = df[(df['Pclass']==3) & (df['Survived']==1)]['PassengerId'].count()

survival_rate = [range_1_survived / range_1, range_2_survived / range_2, range_3_survived / range_3]
print("Hành khách ở hạng vé {} có tỉ lệ sống xót cao nhất".format(survival_rate.index(max(survival_rate))+1))
'''

#Task5
'''
new_df = df.sort_values('Fare', ascending = True)
print(tabulate(new_df, headers='keys', tablefmt='psql'))

range_0_50 = df[(df['Fare']>=0) & (df['Fare']<=50)]['PassengerId'].count()
range_50_100 = df[(df['Fare']>50) & (df['Fare']<=100)]['PassengerId'].count()
range_100_200 = df[(df['Fare']>100) & (df['Fare']<=200)]['PassengerId'].count()
range_200 = df[(df['Fare']>200)]['PassengerId'].count()

range_0_50_Survived = df[(df['Fare']>=0) & (df['Fare']<=50) & (df['Survived']==1)]['PassengerId'].count()
range_50_100_Survived = df[(df['Fare']>50) & (df['Fare']<=100) & (df['Survived']==1)]['PassengerId'].count()
range_100_200_Survived = df[(df['Fare']>100) & (df['Fare']<=200) & (df['Survived']==1)]['PassengerId'].count()
range_200_Survived = df[(df['Fare']>200) & (df['Survived']==1)]['PassengerId'].count()

survival_rate = [
    range_0_50_Survived/range_0_50,
    range_50_100_Survived/range_50_100,
    range_100_200_Survived/range_100_200,
    range_200_Survived/range_200
]

print(*survival_rate)

'''

#Task6
'''
compensation = df[df['Survived']==0]['Fare'].sum()*15
print(compensation)
'''

#Task7
'''
range_men = df[df['Sex']==1]['PassengerId'].count()
range_women = df[df['Sex']==0]['PassengerId'].count()

range_men_survived = df[df['Sex']==1 & (df['Survived']==1)]['PassengerId'].count()
range_women_survived = df[df['Sex']==0 & (df['Survived']==1)]['PassengerId'].count()

survival_rate = [
    range_men_survived/range_men,
    range_women_survived/range_women
]
print('Hanh khach nu co co hoi song xot {} hon nam'.format('cao' if survival_rate[0]<survival_rate[1] else "thap"))
'''

input = df[['Pclass','Sex','Age', 'Fare', 'No_Relatives']]
output = df[['Survived']]

x_train = input[:int(len(input)*0.7)]
y_train = output[:int(len(input)*0.7)]
x_test = input[int(len(input)*0.7):]
y_test = output[int(len(input)*0.7):]

# print(x_train, y_train, x_test, y_test)

scaler = StandardScaler()
print(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
print(x_train)

knn = KNeighborsClassifier(7)
knn.fit(x_train, y_train)
y_predition = knn.predict(x_test)
accuracy = accuracy_score(y_test, y_predition)
print(accuracy)
# print(len(y_test), len(y_predition))

y_temp_pre = knn.predict_proba([[1, 0, 30, 200, 1]])
print(y_temp_pre)
