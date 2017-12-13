from sklearn import datasets
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from MITKnn import Knn
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

clf = Knn()
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pred))

for i in range(len(y_test)):
    print(y_test[i],pred[i])