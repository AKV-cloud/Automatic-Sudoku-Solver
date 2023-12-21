import numpy as np
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import pickle

def mk_dataset(data, target, size):
    indx = np.random.choice(len(target), size, replace=False)
    train_img = data.iloc[indx].to_numpy()
    train_target = target.iloc[indx].to_numpy()
    return train_img, train_target

def skl_knn(k, data, target):
    fifty_x, fifty_y = mk_dataset(data, target, 50000)
    test_img = data.iloc[60000:70000].to_numpy()
    test_target = target.iloc[60000:70000].to_numpy()

    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(fifty_x, fifty_y)

    y_pred = classifier.predict(test_img)
    pickle.dump(classifier, open('knn.sav', 'wb'))
    print(classification_report(test_target, y_pred))
    print("KNN Classifier model saved as knn.sav!")

# Fetch MNIST dataset
mnist = datasets.fetch_openml('mnist_784', data_home='mnist_dataset/')
data, target = mnist.data, mnist.target

# Specify the number of neighbors (k)
k_value = 3

# Call the function to train, test, and save the model
skl_knn(k_value, data, target)