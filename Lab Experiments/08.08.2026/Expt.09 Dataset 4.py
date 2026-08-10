from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
X=[[800],[900],[1000],[1200],[1400],[1600],[1800],[2000],[2200],[2400]]
y=[30,35,40,50,60,75,90,105,120,135]
model=LinearRegression().fit(X,y)
print("Linear:",model.predict([[1700]]))
X2=PolynomialFeatures(2).fit_transform(X)
model2=LinearRegression().fit(X2,y)
print("Poly:",model2.predict(PolynomialFeatures(2).fit_transform([[1700]])))
