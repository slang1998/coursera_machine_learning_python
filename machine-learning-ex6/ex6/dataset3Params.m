function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

c_vec = [0.01 0.03 0.1 0.3 1 3 10 30];
s_vec = [0.01 0.03 0.1 0.3 1 3 10 30];

%c_vec = [0.01 0.03 ];
%s_vec = [0.01 0.03 ];

m = size(c_vec, 2);
n = size(s_vec, 2);
error_vec = zeros(m*n, 3);		% (C, sigma, error)를 행 벡터로 저장한다.

i = 1
for c_i = c_vec;
	for s_i = s_vec;
		% svm을 학습시킨다.
		model= svmTrain(X, y, c_i, @(x1, x2) gaussianKernel(x1, x2, s_i));

		% 학습된 모델과 cross validation set를 이용해서 예측한다.
		predictions = svmPredict(model, Xval);
		err = mean(double(predictions ~= yval));

		% 파라미터와 에러를 매트릭스로 저장한다.
		error_vec(i, :) = [c_i s_i err];
		i = i + 1
	end
end

% 최소 에러를 갖는 C, sigma 값을 구한다.
[mx, mx_index] = min(error_vec(:, 3));

C = error_vec(mx_index, 1);
sigma = error_vec(mx_index, 2);

%error_vec

% =========================================================================

end
