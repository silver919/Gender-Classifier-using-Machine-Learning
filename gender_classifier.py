from sklearn import tree
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import numpy

#[height,weight,shoe size]
X = [[181,71,44], [151,44,38], [188,85,47], [160,47,38], [144,30,38],
	 [190,89,44], [182,71,42], [150,42,38], [153,46,36], [184,72,40],
	 [180,79,48]]

Y = ['male','female','male','female','female','male','male',
	 'female','female','male','male']

#classifiers
clf_tree = tree.DecisionTreeClassifier()
clf_neighbor = neighbors.KNeighborsClassifier()
clf_gaussian = naive_bayes.GaussianNB()
clf_percep = linear_model.Perceptron()

#Training the models
clf_tree = clf_tree.fit(X,Y)
clf_neighbor = clf_neighbor.fit(X,Y)
clf_gaussian = clf_gaussian.fit(X,Y)
clf_percep = clf_percep.fit(X,Y)

#Testing on the test data and calculating the accuracy score

_X=[[184,84,44],[198,92,48],[183,83,44],[166,47,36],[170,60,38],[172,64,39],[182,80,42],[180,80,43]]
_Y=['male','male','male','female','female','female','male','male']

pred_tree = clf_tree.predict(_X)
acc_tree = accuracy_score(_Y, pred_tree)*100
print('Accuracy of Decision Tree: {}'.format(acc_tree) + '%')

pred_neighbor = clf_neighbor.predict(_X)
acc_neighbor = accuracy_score(_Y,pred_neighbor)*100
print('Accuracy of K Neighbor Classifier: {}'.format(acc_neighbor) + '%')

pred_gaussian = clf_gaussian.predict(_X)
acc_gaussian = accuracy_score(_Y, pred_gaussian)*100
print('Accuracy of GaussianNB: {}'.format(acc_gaussian) + '%')

pred_percep = clf_percep.predict(_X)
acc_percep = accuracy_score(_Y, pred_percep) * 100
print('Accuracy of Perceptron: {}'.format(acc_percep) + '%')

#find the best classifier among them
i = numpy.argmax([acc_tree, acc_neighbor, acc_gaussian, acc_percep])
classifier = {0: 'Decision Tree', 1: 'K Neighbors Classifier', 2: 'Naive bayes', 3: 'Perceptron'}
print('The best classifier among them is {}',format(classifier[i]))



