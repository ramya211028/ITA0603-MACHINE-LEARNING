from sklearn.tree import DecisionTreeClassifier
X = [
    [2,2,1,2],
    [2,3,1,3],
    [1,2,1,2],
    [1,1,0,1],
    [0,0,0,0],
    [2,2,0,2],
    [0,1,0,1],
    [1,2,1,3],
    [2,3,1,2],
    [0,0,1,1]
]
y = ['Yes','Yes','Yes','No','No','Yes','No','Yes','Yes','No']
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X,y)
print(clf.predict([[2,2,1,3]]))
