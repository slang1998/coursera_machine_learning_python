def computeCost(X, y, theta):    
    m = np.size(y)
    
    J = 0
    # X: 97x2, theta: 2x1
    predictions = X * theta
    
    # nx1 의 컬럼 벡터를 제곱할 때 아래와 같이 한다.
    sqrErrors = np.asarray(predictions - y) ** 2
    J = 1 / (2 * m) * np.sum(sqrErrors)
    
    return J
