function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;							% m x 2 dimensions
n = size(X, 2);
%mu = zeros(1, size(X, n));			% 1 x 2 dimensions
%sigma = zeros(1, size(X, n));		% 1 x 2 dimensions
mu = zeros(1, n);			% 1 x 2 dimensions
sigma = zeros(1, n);		% 1 x 2 dimensions

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

m = size(X, 1);
mu = mean(X_norm);
sigma = std(X_norm);

% repmat: mu 행렬을 m, 1 만큼 반복해서 생성한다.
X_norm = (X_norm .- repmat(mu, m, 1)) ./ repmat(sigma, m, 1);


% ============================================================

end
