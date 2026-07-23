from sklearn.neural_network import MLPClassifier
X = [
    [1,1,1,1,1],
    [1,1,0,1,1],
    [0,1,1,0,0],
    [1,0,1,1,1],
    [0,0,0,0,0],
    [1,1,1,0,1],
    [0,1,0,1,0],
    [1,0,0,1,1],
    [1,1,1,1,1],
    [0,0,1,0,0]
]
y = [1,1,0,1,0,1,0,1,1,0]
ann = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000)
ann.fit(X, y)
print("Prediction:", ann.predict([[1,1,0,1,1]]))
