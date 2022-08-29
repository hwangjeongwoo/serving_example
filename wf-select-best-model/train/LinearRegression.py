from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LinearRegression

def get_model(X_train, Y_train):
    steps = [
        ('scalar', MinMaxScaler()),
        ('poly', PolynomialFeatures(degree=1)),
        ('model', LinearRegression())
    ]
    lr_pipe = Pipeline(steps)
    lr_pipe.fit(X_train, Y_train)
    return lr_pipe
