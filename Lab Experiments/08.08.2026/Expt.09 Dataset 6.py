from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
X=[[40],[50],[60],[70],[80],[90],[100],[110],[120],[130]]
y=[4.5,5.0,5.6,6.5,7.8,9.2,10.8,12.5,14.3,16.2]
model=LinearRegression().fit(X,y)
print("Linear:",model.predict([[130]]))
X2=PolynomialFeatures(2).fit_transform(X)
model2=LinearRegression().fit(X2,y)
print("Poly:",model2.predict(PolynomialFeatures(2).fit_transform([[130]])))
