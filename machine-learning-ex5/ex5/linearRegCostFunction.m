function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% input
% X: 12x2
% theta: 2x1

% 1.2 Regulaized linear regression cost 구하기
% ----------------------------------------------------
predictions = X * theta;
sqrErros = (predictions - y).^2;

J = 1/(2 * m) * sum(sqrErros);

% regularization term 값을 구한다.
sum_theta_square = sum(sum(theta(2:end) .^ 2));
reg_term = lambda / (2 * m) * sum_theta_square;

% reg_term을 합한다.
J = J + reg_term;



% 1.3 Regulaized linear regression gradient 구하기
% ----------------------------------------------------

% theta_0 에 대한 편미분값을 구한다. delta_0 는 scalar 값이다.
predictions = X * theta;           			% predictions: mx1 dimensions
errors = (predictions - y);                 % error: mx1 dimensions
delta_0 = 1/m * errors' * X(:,1);           % X의 첫번째 컬럼(bias) 에 대해서 계산  

% theta_j 에 대한 편미분값을 구한다.
delta = 1/m * errors' * X(:,2:end) ;        % delta: 1x27  
regular = lambda / m * theta(2:end);		% regular: 27x1
delta_j = delta' + regular;					% delta_j: 27x1

grad = [delta_0, delta_j'];



% =========================================================================

grad = grad(:);

end
