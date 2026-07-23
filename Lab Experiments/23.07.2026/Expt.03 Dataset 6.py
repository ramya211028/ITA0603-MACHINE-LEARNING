from sklearn.tree import DecisionTreeClassifier
X = [
    [2,3,1,1],
    [2,2,1,1],
    [1,2,0,1],
    [0,1,0,0],
    [1,3,1,1],
    [0,0,0,0],
    [2,2,1,0],
    [1,1,0,1],
    [2,3,1,1],
    [0,0,0,1]
]
y = ['Yes','Yes','Yes','No','Yes','No','Yes','No','Yes','No']
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X,y)
print(clf.predict([[1,2,1,1]]))
