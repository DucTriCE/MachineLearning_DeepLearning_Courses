import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.svm import SVC

data = pd.read_csv('diabetes.csv', sep=',')

    #Count each of the outcome
# print(data['Outcome'].value_counts())
###Using matplotlib
# plt.figure(figsize=(8,8))
# sn.histplot(data['Outcome'])
# plt.title("Distribution")
# plt.show()

    #Heatmap
# sn.heatmap(data.corr(), annot=True)
# plt.show()

    #Divide
target = 'Outcome'
x = data.drop(target, axis = 1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

    #Build model
clf = SVC()
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
print(classification_report(y_test,y_predict))