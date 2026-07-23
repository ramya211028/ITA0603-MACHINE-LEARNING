from sklearn.tree import DecisionTreeClassifier
X = [
    [1,1,1,1],
    [1,1,0,1],
    [0,1,1,0],
    [1,0,1,1],
    [0,0,0,0],
    [1,1,1,0],
    [0,1,0,1],
    [1,0,0,1],
    [1,1,1,1],
    [0,0,1,0]
]
y = ['Positive','Positive','Negative','Positive','Negative',
     'Positive','Negative','Positive','Positive','Negative']
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X,y)
print(clf.predict([[1,1,0,1]]))
