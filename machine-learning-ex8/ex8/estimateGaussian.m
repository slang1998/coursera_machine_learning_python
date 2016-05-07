function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

% 컬럼 벡터 기준으로 평균을 계산한다.
% 참고: mean(X, 2) 는 행 기준으로 평균을 계산한다.
mu = mean(X)

% var() 함수는 1 / (m - 1) 을 사용한다.
% 1/m 으로 만들기 위해 (m - 1) / m을 곱해준다.
sigma2 = (m - 1) / m * var(X)





% =============================================================


end
