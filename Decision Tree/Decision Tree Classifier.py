#Imports Graphviz for Tree Viewing
import graphviz
#Imports iris dataset & tree from sklearn
from sklearn.datasets import load_iris
from sklearn import tree
#Loads Dataset
iris = load_iris()
#Loads a clf object with Decision Tree Classifier
clf = tree.DecisionTreeClassifier()
#Classifier is trained using Fit() funct. with dataset
clf = clf.fit(iris.data, iris.target)
#Predicts on Test Set using predict function which takes input of test eg
prediction = clf.predict(iris.data[:1, :])
#Outputs Prediction probablity for each class in classifier
probablity = clf.predict_proba(iris.data[:1, :])
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")
print("Prediction On Test Set\n")
print(prediction, "\nProbablity of Every Class Predicted \n", probablity)
#Graph shows the tree in Graphical Manner
graph