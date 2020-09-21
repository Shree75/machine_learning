import pandas as pd
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

data = pd.read_csv('car.data')

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['buying']))
door = le.fit_transform(list(data['maint']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
clas = le.fit_transform(list(data['class']))

predict = 'class'

X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(clas)

x_test, x_train, y_test, y_train = sklearn.model_selection.train_test_split(X, y, test_size = 0.2)

model = KNeighborsClassifier(n_neighbors = 5)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test )

names = ["unacc", "acc", "good", "vgood"]

for i in range(len(x_test)):
	print("Predicted: ", names[predicted[i]],"Data: ", x_test[i], "Actual: ",names[y_test[i]])
	n = model.kneighbors([x_test[i]], 4, True)
	print("N: ", n)