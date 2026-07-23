from sklearn.tree import DecisionTreeClassifier
X = [
    [0,0,0,0],
    [0,0,0,1],
    [1,0,0,0],
    [2,1,0,0],
    [2,2,1,0],
    [2,2,1,1],
    [1,2,1,1],
    [0,1,0,0],
    [0,2,1,0],
    [2,1,1,0],
    [0,1,1,1],
    [1,1,0,1],
    [1,0,1,0],
    [2,1,0,1]
]
y = ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X,y)
print(clf.predict([[0,2,0,0]]))
