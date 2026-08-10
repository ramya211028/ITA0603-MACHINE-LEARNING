from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
X=[[2],[3],[4],[5],[6],[8],[10],[12],[14],[16]]
y=[20,25,30,40,50,65,75,82,90,96]
model=LinearRegression().fit(X,y)
print("Linear:",model.predict([[9]]))
X2=PolynomialFeatures(2).fit_transform(X)
model2=LinearRegression().fit(X2,y)
print("Poly:",model2.predict(PolynomialFeatures(2).fit_transform([[9]])))
