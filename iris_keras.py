# import matplotlib
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# iris = sns.load_dataset("iris")
iris = load_iris()
# sns.pairplot(iris.data, hue='species')
# plt.show()
X = iris.data[:, 0:4]
y = iris.target

train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.5, random_state=0)
# lr = LogisticRegressionCV(max_iter=7600)
# lr.fit(train_X, train_y)
# pred_y = lr.predict(test_X)
#
# print(f"Test fraction correct (Accuracy) = {lr.score(test_X, test_y):.2f}")

# def one_hot_encode_object_array(arr):
#     '''One hot encode a numpy array of objects (e.g. strings)'''
#     uniques, ids = np.unique(arr, return_inverse=True)
#     return np_utils.to_categorical(ids, len(uniques))
#
# train_y_ohe = one_hot_encode_object_array(train_y)
# test_y_ohe = one_hot_encode_object_array(test_y)

model = Sequential()
model.add(Dense(4, input_dim=4, activation='relu'))
# model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# estimator = KerasClassifier(build_fn=model, nb_epoch=150, batch_size=5, verbose=1)
# kfold = KFold(n_splits=10, shuffle=True)
# results = cross_val_score(estimator, train_X, train_y_ohe, cv=kfold)

model.fit(train_X, train_y, verbose=0, batch_size=75, epochs=300)
loss, accuracy = model.evaluate(test_X, test_y, verbose=1)
print(f"Test fraction correct (Accuracy) = {accuracy:.2f}")
