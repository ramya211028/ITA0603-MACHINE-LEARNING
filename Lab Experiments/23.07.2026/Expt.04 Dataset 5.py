from sklearn.neural_network import MLPClassifier
X = [
    [2,3,1,1],
    [2,2,1,1],
    [1,2,1,1],
    [1,1,0,1],
    [0,0,0,0],
    [2,3,1,0],
    [1,2,0,1],
    [0,1,0,0],
    [2,2,1,1],
    [0,0,0,1]
]
y = [1,1,1,0,0,1,1,0,1,0]
ann = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000)
ann.fit(X, y)
print("Prediction:", ann.predict([[1,2,1,1]]))
