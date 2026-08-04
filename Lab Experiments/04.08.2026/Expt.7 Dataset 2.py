from sklearn.linear_model import LinearRegression
X=[[800,2,10],[1000,2,8],[1200,3,6],[1400,3,5],
   [1600,4,4],[1800,4,3],[2000,5,2],[2200,5,1],
   [2400,6,1],[2600,6,0]]
y=[35,45,55,65,75,85,95,105,115,125]
m=LinearRegression()
m.fit(X,y)
print("Prediction:",m.predict([[1700,4,3]]))
