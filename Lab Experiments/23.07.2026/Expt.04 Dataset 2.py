from sklearn.neural_network import MLPClassifier
X = [
    [9,3,1,3],
    [8,2,1,2],
    [7,2,1,1],
    [6,1,0,1],
    [5,0,0,0],
    [9,3,1,2],
    [8,2,0,2],
    [6,1,1,1],
    [7,3,1,2],
    [5,0,0,1]
]
y = [1,1,1,0,0,1,1,0,1,0]
ann = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000)
ann.fit(X, y)
print("Prediction:", ann.predict([[8,3,1,2]]))
