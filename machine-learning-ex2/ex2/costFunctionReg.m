function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

% X: 118 x 28
n = size(theta);	% n is size of theta_0 ~ theta_n


% 1) cost J(theta) 값을 구한다.
% -------------------------------------------------------
% 벡터 * 행렬은 .* 를 사용해서 요소 곱으로 해야 한다.
% theta를 theta_0와 그 나머지 벡터로 분리한다.
theta_0 = theta(1);
theta_j = theta(2:n);


sum_theta_square = sum(theta_j .^ 2);

J = 1/m * sum(-y .* log(sigmoid(X * theta)) - (1 - y) .* log (1 - sigmoid(X * theta))) ...
	+ lambda / (2 * m) * sum_theta_square ;


% 2) theta 편미분 값을 구한다.
% -------------------------------------------------------
%grad = 1/m * sum( (sigmoid(X * theta) - y) ) ;

% theta_0 에 대한 편미분값을 구한다.
predictions = sigmoid(X * theta);           % predictions: mx1 dimensions
errors = (predictions - y);          		% error: mx1 dimensions
% size(errors)

delta_0 = 1/m * errors' * X(:,1);         	% X의 첫번째 컬럼(bias) 에 대해서 계산  


% theta_j 에 대한 편미분값을 구한다.
% X의 첫번째 컬럼 삭제한다. 첫번째 컬럼값은 theta_0에 대한 내용이기 때문이다.
X(:, 1) = [];								

%delta = 1/m * errors' * X + lambda / m * theta_j ;           %   
delta = 1/m * errors' * X ;           	% delta: 1x27
regular = lambda / m * theta_j;			% regular: 27x1


delta_j = delta' + regular;

grad = [delta_0, delta_j']';





% =============================================================

end
