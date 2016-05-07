function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% theta : (n+1) x 1
% X: m x (n+1)
% y: m x 1
% .* 를 사용해서 요소 곱으로 해야 한다.

J = 1/m * sum(-y .* log(sigmoid(X * theta)) - (1 - y) .* log (1 - sigmoid(X * theta)));


%grad = 1/m * sum( (sigmoid(X * theta) - y) ) ;

predictions = sigmoid(X * theta);            % predictions: mx1 dimensions
errors = (predictions - y);          % error: mx1 dimensions

delta = 1/m * errors' * X;           %   
grad = delta';



% =============================================================

end
