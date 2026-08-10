from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
X=[[2],[3],[4],[5],[6],[7],[8],[9],[10],[11]]
y=[65,70,75,80,85,88,90,93,95,98]
model=LinearRegression().fit(X,y)
print("Linear:",model.predict([[7]]))
X2=PolynomialFeatures(2).fit_transform(X)
model2=LinearRegression().fit(X2,y)
print("Poly:",model2.predict(PolynomialFeatures(2).fit_transform([[7]])))
