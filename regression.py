import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1","G2","G3","studytime","failures","absences"]]

predict = "G3"

X = np.array(data.drop([predict], 1))

y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

# best = 0
# for x in range(30):
# 	#x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

# 	linear = linear_model.LinearRegression()

# 	linear.fit(x_train, y_train)
# 	accuracy = linear.score(x_test, y_test)
# 	#acc = accuracy*100
# 	print(accuracy, "%")

# 	if accuracy > best:
# 		best = accuracy
# 		with open("studentmodel.pickle", "wb") as f:
# 			pickle.dump(linear, f)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Coefficient: \n" , linear.coef_ )
print("Intercept: \n" , linear.intercept_ )

prediction = linear.predict(x_test)

for i in range(len(prediction)):
	print(prediction[i], x_test[i], y_test[i])

p = 'G1'
style.use("ggplot")
plt.scatter(data[p], data["G3"])
#plt.plot(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()




