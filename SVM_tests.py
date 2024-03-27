# Testing
if __name__=="main":
    from sklearn import datasets
    import matplotlib.pyplot as plt
    import numpy as np 
    X,y= datasets.make_blobs(
        nb_samples=50, nb_features=3,centers=3,cluster_std=1.08, random_state=30
    )
    y = np.where(y==0,-1,1)
