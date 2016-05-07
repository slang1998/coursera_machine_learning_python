function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% X: 300x2

m = size(X, 1);
mat_dist = zeros(m, K);				% 각 클러스터 중심점과의 거리값 저장


for i = 1:K
	v = centroids(i, :);			% 하나의 클러스터 중심에 대해서
	diff = X - repmat(v, m, 1);		% X 벡터와의 거리 차이를 구하고
	dist = sum( diff .^ 2, 2);		% 거리값을 구한 다음
	mat_dist(:, i) = dist;			% mat_dist에 저장한다.
end

[val idx] = min(mat_dist, [], 2);	% 행 기준으로 최소값의 인덱스를 찾는다.


% =============================================================

end

