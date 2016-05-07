function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% X * Theta' 는 예측값이다.
% R .* Y 은 사용자의 평가 접수가 있을 경우(R) 해당 영화에 대한 평가점수 이다.(Y)
J = 1 / 2 * sum(sum( (( (X * Theta') - Y) .* R).^2 ));


% 영화 i번째에 대한 피쳐 x를 생각할 때, 영화를 평가한 사용자들만 생각하면 된다.
% 방법으로, 영화 i에 대해서 x(i)_1,,,,x(i)_n 에 대한 모든 피쳐의 미분을 한번에 계산한다.
% 	Theta, Y에서 관심있는 항목만 추출한다. (r(i, j) = 1)

% 모든 영화에 대해서 loop를 실행한다.
for i = 1:num_movies

	% 영화 i를 평가한 사용자의 인덱스 정보를 추출한다.
	idx = find(R(i, :)==1);

	% 인덱스 정보를 이용해서 영화를 평가한 실제 사용자의 theta와 평가점수를 구한다.
	Theta_temp = Theta(idx, :);
	Y_temp = Y(i, idx);				% 1x2 dimension

	% 영화에 대한 피쳐 2개가 계산된다. X_grad(i, :) = 1x2 dimension
	X_grad(i, :) = (X(i, :) * Theta_temp' - Y_temp) * Theta_temp;

	% 2.2.4 Regularized gradient를 계산한다.
	X_grad(i, :) = X_grad(i, :) + lambda * X(i, :);

end

% 모든 사용자에 대해서 loop를 실행한다.
for j = 1:num_users

	% 사용자 j가 평가한 영화의 인덱스 정보를 추출한다.
	idx = find(R(:, j) == 1);	% idx는 몇 번째 영화인지를 나타낸다.

	% 인덱스 정보를 이용해서 영화를 평가한 실제 사용자의 theta와 평가점수를 구한다.
	Theta_temp = Theta(j, :);
	Y_temp = Y(idx, j);

	X_temp = X(idx, :);

	% Theta_grad(j, :) = 1x2 dimension
	Theta_grad(j, :) = (X_temp * Theta_temp' - Y_temp)' * X_temp;
	
	% 2.2.4 Regularized gradient를 계산한다.
	Theta_grad(j, :) = Theta_grad(j, :) + lambda * Theta(j, :);

end

% 2.2.3 Regularized cost function 구현
J = J + (lambda / 2) * sum(sum(Theta.^2)) + (lambda / 2) * sum(sum(X.^2));


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
