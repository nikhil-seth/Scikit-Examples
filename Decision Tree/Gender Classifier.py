#Gender Classifier Using Decision Tree
#Importing Tree
from sklearn import tree
#Making X & Y as Training Set with 3 features & 11 Examples
#X Length,Width,Shoe Size
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39], [171, 75, 42], [177, 70, 40], [159, 55, 37],
     [181, 85, 43]]
#Y : Result of 11 Examples
Y = [
    'male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
    'female', 'male', 'male'
]
#clf object of DecisionTreeClassifier
clf = tree.DecisionTreeClassifier()
#fit() function used to train clf
clf = clf.fit(X, Y)
#prediction from Trained Classifier
prediction = clf.predict([[190, 70, 43]])
print("Prediction \nLength :190\nWidth :70\nShoe Size :43\nResult By Model : ",
      prediction)