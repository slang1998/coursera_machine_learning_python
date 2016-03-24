def gradientDescent(X, y, theta, alpha, num_iters):
    m = size(y)
    J_history = np.zeros((num_iters, 1))
    
    for iter in range(num_iters):
        
        predictions = matrix(X) * matrix(theta)
        errors = predictions - y
        
        delta = (1 / m) * errors.T * X
        theta = theta - alpha * delta.T
        
        J_history[iter, 0] = computeCost(X, y, theta)
        
    return theta, J_history
