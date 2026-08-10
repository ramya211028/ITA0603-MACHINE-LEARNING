from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
X=[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
y=[3.0,3.8,4.8,6.0,7.5,9.0,10.8,12.6,14.5,16.5]
model=LinearRegression().fit(X,y)
print("Linear:",model.predict([[7]]))
X2=PolynomialFeatures(2).fit_transform(X)
model2=LinearRegression().fit(X2,y)
print("Poly:",model2.predict(PolynomialFeatures(2).fit_transform([[7]])))
