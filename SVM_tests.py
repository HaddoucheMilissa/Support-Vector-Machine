# Testing
if __name__=="main":
    from sklearn import datasets
    import matplotlib.pyplot as plt
    import numpy as np 
    from sklearn.svm import SVC 
    X,y= datasets.make_blobs(
        nb_samples=50, nb_features=3,centers=3,cluster_std=1.08, random_state=30
    )
    x,y =np.where(y==0,-1,1)
classifier1=SVM()
classifier1.fit(x,y)
print(classifier1.w,classifier1.b)

def visualize_svm():
    def get_hyperplane_value(x,w,b,offset):
            return (-w[0]*x + b + offset)/w[2]
    fig=plt.figure()
    ax=fig.add_subplot(1, 1, 1)
    plt.scatter(X[:,0], X[:,2], marker="o",c=y)