import urllib
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import numpy as np
import tensorflow as tf
import urllib.request

def get_proper_images(raw):
    raw_float = np.array(raw, dtype=float) 
    images = raw_float.reshape([-1, 3, 32, 32])
    images = images.transpose([0, 2, 3, 1])
    return images


def onehot_labels(labels):
    return np.eye(100)[labels]


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    

X = get_proper_images(unpickle('train')[b'data'])
Y = onehot_labels(unpickle('train')[b'fine_labels'])
X_test = get_proper_images(unpickle('test')[b'data'])
Y_test = onehot_labels(unpickle('test')[b'fine_labels'])

X_ = unpickle('train')[b'data']
Y_ = unpickle('train')[b'fine_labels']
X_test_ = unpickle('test')[b'data']
Y_test_ = unpickle('test')[b'fine_labels']

#==============================================================================
# nsamples, nx, ny = X.shape
# X = X.reshape((nsamples,nx*ny))
#  
# nsamples, nx, ny = Y.shape
# Y = Y.reshape((nsamples,nx*ny))
#  
# nsamples, nx, ny = X_test.shape
# X_test = X_test.reshape((nsamples,nx*ny))
#  
# nsamples, nx, ny = Y_test.shape
# Y_test = Y_test.reshape((nsamples,nx*ny))
# 
#==============================================================================
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=15.)
network = input_data(shape=[None, 32, 32, 3],
                       data_preprocessing=img_prep,
                       data_augmentation=img_aug)
network = conv_2d(network, 32, 3, strides=1, padding='same', activation='relu', bias=True, 
                    bias_init='zeros', weights_init='uniform_scaling')
network = max_pool_2d(network, 2 , strides=None, padding='same')
network = conv_2d(network, 64, 3, strides=1, padding='same', activation='relu', bias=True, 
                    bias_init='zeros', weights_init='uniform_scaling')
network = conv_2d(network, 64, 3 , strides=1, padding='same', activation='relu', bias=True, 
                    bias_init='zeros', weights_init='uniform_scaling')
network = max_pool_2d(network, 2 , strides=None, padding='same')
network = fully_connected(network, 600, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 100, activation='softmax')
network = regression(network, optimizer='adam',
                       loss='categorical_crossentropy',
                       learning_rate=0.001)
                       
#==============================================================================
# with tf.device('cpu:0'):
#       model = tflearn.DNN(network, tensorboard_verbose=0)
#       model.fit(X, Y, n_epoch=3, shuffle=True, validation_set=(X_test, Y_test), show_metric=True, batch_size=100 , run_id='aa2')
#   
#==============================================================================
 

#==============================================================================
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
clf = RandomForestClassifier()
neigh= BaggingClassifier()
ada = AdaBoostClassifier()
clf.fit(X_, Y_)
pred = clf.predict(X_test_)
neigh.fit(X_,Y_)
pred_b = neigh.predict(X_test_)
ada.fit(X_,Y_)
pred_a = ada.predict(X_test_)
 
print(accuracy_score(Y_test_,pred))
print(accuracy_score(Y_test_,pred_b))
print(accuracy_score(Y_test_,pred_a))
