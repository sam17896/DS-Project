# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 00:43:18 2018

@author: HP
"""

import numpy as np
import data_utils as d
from sklearn.metrics import accuracy_score as acc
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import threading
import matplotlib.pyplot as plt

X_train, y_train, X_test, y_test = d.load_CIFAR10('cifar-10-batches-py')
num_training = 5000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print (X_train.shape, X_test.shape)

#K-NN
def knn_thread():
    x = range(1, 10)
    y = []
    def knn(k):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        y.append(acc(y_test, pred))
    
    threads = []
    j = 0
    for i in x:
        threads.append(threading.Thread(target=knn, args=(i,)))
        threads[j].start()
        j += 1
        
    j = 0
    for i in x:
        threads[j].join()
        j += 1
    
    plt.plot(x, y, color='red', label='K-Nearest Neighbor')

def decision_tree_thread():
    x = range(1, 10)
    y = []
    def decision_tree(seed):
        clf = DecisionTreeClassifier(random_state=seed)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        y.append(acc(y_test, pred))
    
    threads = []
    j = 0
    for i in x:
        threads.append(threading.Thread(target=decision_tree, args=(i,)))
        threads[j].start()
        j += 1
        
    j = 0
    for i in x:
        threads[j].join()
        j += 1
    
    plt.plot(x, y, color='green', label='Decision-Tree')


t1 = threading.Thread(target=knn_thread)
t2 = threading.Thread(target=decision_tree_thread)
t1.start()
t2.start()
t1.join()
t2.join()

plt.ylabel('Accuracy')
plt.legend()
plt.savefig('comparison.png')
plt.show()