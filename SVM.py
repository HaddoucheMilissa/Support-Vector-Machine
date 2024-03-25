from sklearn.datasets import loadbreast_cancer # importing data
from sklearn.model_selection import train_test_split # split the data
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier

data=load_breast_cancer
x=data.data
y=data.target
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
#1st classifier
classifier1=SVC(kernel='linear',C=5 )
classifier1.fit(x_train,y_train)


#result is : 0.956140350877193
