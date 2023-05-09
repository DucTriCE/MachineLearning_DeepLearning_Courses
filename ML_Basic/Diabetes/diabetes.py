from tabulate import tabulate
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv(filepath_or_buffer="data/diabetes_2.csv", header=0)
# print(tabulate(df,headers='keys', tablefmt="psql"))

'''
#Check Datashape

print ("Shape of data {}" . format (df.shape))
print ("Number of rows: {}" . format (df.shape[0]))
print ("Number of columns: {}" . format (df.shape[1]))
print("\n") 
print(df.info())


#Take out training and testing features

input = df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
output = df.Outcome

x_train = input[:int(0.7*len(input))]
y_train = output[:int(0.7*len(output))]
x_test = input[int(0.7*len(output)):]
y_test = output[int(0.7*len(output)):]

# print(x_train, y_train, x_test, y_test)

#Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_train)

knn = KNeighborsClassifier(3)
knn.fit(x_train,y_train)
y_prediction = knn.predict(x_test)
accuracy = accuracy_score(y_test, y_prediction)
print(accuracy)

'''

#Đếm số người phụ nữ bị tiểu đường group theo pregnancies
new_df = df[df['Outcome']==1].groupby('Pregnancies')['Pregnancies'].count().reset_index(name='Count')
# print(new_df)

#Đếm tổng số người phụ nữ group theo pregnancies
new_df_2 = df.groupby('Pregnancies')['Pregnancies'].count().reset_index(name='Count')
# print(new_df_2)

#Cái này là ra cột xác suất bị tiểu đường ( tên của cột đó là Count, đổi được ),
#Còn pregnancies = 1.0 là do lấy 2 dataframe chia nhau nên nó = 1, có thể fix được
print(new_df['Count']/new_df_2['Count'])

