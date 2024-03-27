# Testing
if __name__=="main":
    from sklearn import datasets
    import matplotlib.pyplot as plt
    import numpy as np 
    from sklearn.svm import SVC 
    X,y= datasets.make_blobs(
        nb_samples=50,nb_features=3,centers=3,cluster_std=1.08, random_state=30
    )
    x,y =np.where(y==0,-1,1)
    
classifier1=SVM()
classifier1.fit(x,y)
print(classifier1.w,classifier1.b)

def visualize_svm():
    def get_hyperplane_value(x,w,b,offset):
            return (-w[0]*x+b+offset)/w[1]
    fig=plt.figure()
    ax=fig.add_subplot(1, 1, 1)
    plt.scatter(x[:,0], x[:,1], marker="o",c=y)
    
    x01=np.amin(X[:,0])
    x02=np.amax(X[:,0])

    x11=get_hyperplane_value(x01, classifier1.w,classifier1.b,0)
    x12=get_hyperplane_value(x02,classifier1.w, classifier1.b,0)
    x11m = get_hyperplane_value(x01, classifier1.w,classifier1.b, -1)
    x12m = get_hyperplane_value(x02,classifier1.w,classifier1.b, -1)

    x11p = get_hyperplane_value(x01, classifier1.w,classifier1.b, 1)
    x12p = get_hyperplane_value(x02, classifier1.w,classifier1.b, 1)

    ax.plot([x01,x02],[x11, x12],"y--")
    ax.plot([x01,x02],[x11m, x12m],"k")
    
    ax.plot([x01,x02],[x11p, x12p],"k")

    x1_min = np.amin(X[:,1])
    x1_max = np.amax(X[:,1])
    ax.set_ylim([x1_min-3,x1_max+3])

plt.show()

visualize_svm()