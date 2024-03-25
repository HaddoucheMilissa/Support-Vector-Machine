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
#create 2nd classifier

classifier2=KNeighborsClassifier(n_neighbors=4)
classifier2.fit(x_train,y_train)
print("Score of the 1st classifier(SVC) :",classifier1.score(x_test,y_test)) # Score of the 1st classifier(SVC) : 0.9210526315789473
print("Score of the 2nd classifier(SVC) :",classifier2.score(x_test,y_test)) # Score of the 2nd classifier(SVC) : 0.9385964912280702

#SVM as a class

import numpy as np
class SVM:
    def __init__(self,learning_rate=0.001,lambda_para=0.01,nb_iterations=1000):
        self.lr=learning_rate
        self.lambda_para= lambda_para
        self.nb_iterations=nb_iterations
        self.w=None
        self.b=None
        
    def fit(self,x,y):
        yy=np.where(y<=0,-1,1)
        nb_samples,nb_features=x.shape
        # nb rows = nb samples
        # nb columns = nb features
        self.w=np.zeros(nb_features)
        self.b=0
        for i in range(self.nb_iterations):
              for index, x_index in enumerate(x):
                  cond= y [index]* (np.dot(x_index,self.w)-self.b)>=1
                  if cond:
                        self.w -= self.lr (2*self.lambda_para *self.w )