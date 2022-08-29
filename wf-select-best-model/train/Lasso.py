from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import Lasso

def get_model(X_train, Y_train):
    steps = [
        ('scalar', MinMaxScaler()),
        ('poly', PolynomialFeatures(degree=2)),
        ('model', Lasso(alpha=0.012, fit_intercept=True, max_iter=3000))
    ]
    lasso_pipe = Pipeline(steps)
    lasso_pipe.fit(X_train, Y_train)
    return lasso_pipe
