function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------

%input:
%[error_train, error_val] = learningCurve( [1 2; 1 3; 1 4; 1 5], [7;6;5;4], [1 7; 1 -2;], [2; 12], 7 )

%output:
%error_train =
%   0.00000
%   0.10889
%   0.20165
%   0.21267
%error_val =
%   12.5000
%   11.1700
%    8.3951
%    5.4696

% X: bias �÷��� �߰��� ���·� �Լ��� ȣ��ȴ�. (ex5.m ����)

for i = 1:m
	% train set�� ���Ѵ�.
	X_train = X(1:i, :);
	y_train = y(1:i);


	% train ��ų���� �Լ��� ���޵� �Ķ���� lambda�� �״�� ����Ѵ�.
	% NOTE: lambda = 0���� �����ϸ� ���� �ȵȴ�!
	[theta] = trainLinearReg(X_train, y_train, lambda);

	% regularization term�� ū �ǹ̰� ��� lambda = 0�� �����ϰ� �Լ� ȣ���Ѵ�.
	% train data set������ cost J�� ���Ѵ�.
	J = linearRegCostFunction(X_train, y_train, theta, 0);
	error_train(i) = J;

	% cross validation set ������ cost J�� ���Ѵ�.
	J = linearRegCostFunction(Xval, yval, theta, 0);
	error_val(i) = J;


end





% -------------------------------------------------------------

% =========================================================================

end
