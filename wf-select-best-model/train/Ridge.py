from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import Ridge

def get_model(X_train, Y_train):
    steps = [
        ('scalar', MinMaxScaler()),
        ('poly', PolynomialFeatures(degree=2)),
        ('model', Ridge(alpha=3.8, fit_intercept=True))
    ]
    ridge_pipe = Pipeline(steps)
    ridge_pipe.fit(X_train, Y_train)
    return ridge_pipe
