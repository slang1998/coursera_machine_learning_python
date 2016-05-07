function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    % m = 97, n = 1
    % X: mx(n+1) dimension
    % theta: (n+1)x1 dimension
    % erros: mx1 dimension
    % delta: (n+1)x1 dimension

    predictions = X * theta;            % predictions: mx1 dimensions
    erros = (predictions - y);          % error: mx1 dimensions

    delta = 1/m * erros' * X;           %   
    theta = theta - alpha * delta';     %   

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

    % for debugging
    % J_history(iter)


end

end
