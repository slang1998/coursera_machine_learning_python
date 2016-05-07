function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% reference:
% test case: https://www.coursera.org/learn/machine-learning/module/Aah2H/discussions/uPd5FJqnEeWWpRIGHRsuuw
% tutorial: https://www.coursera.org/learn/machine-learning/discussions/a8Kce_WxEeS16yIACyoj1Q

% PART 1: 
% ================
% input layer: 400
% hidden layer: 25
% output layer: 10
% X: 5000x400 
% a1: 5000x401 => x항에 bias 추가됨
% z2: 5000x25 => a1 * Theta1' = 5000x401 * (25x401)' = 5000x25
% a2: 5000x26 => a2 항에 bias 추가됨
% z3: 5000x10 => a2 * Theta2' = 5000x26 * (10x26)' = 5000x10
% a3: 5000x10 => sigmoid(z3)
% d3: 5000x10
% d2: 5000x25
% Theta1, Delta1 and Theta1_grad: 25x401
% Theta2, Delta2 and Theta2_grad: 10x26

% 1) 1의 컬럼벡터 추가, m: size of X
% ---------------------------------------
a1 = [ones(m, 1) X];
X = a1;

% 2) z 계산 시 theta x X' 형식을 취한다.
% ---------------------------------------
% a2의 출력값 계산
z2 = a1 * Theta1';		% 5000x401 * (25x401)'
a2 = sigmoid(z2);
% z2, a2: 5000x25


% 3) 다음 단계의 hidden layer에 대한 입력은 mxn 형태로 만든다.
% ---------------------------------------
% Add ones to the a2 data matrix
a2 = [ones(size(a2, 1), 1) a2];
z3 = a2 * Theta2' ; 	% 5000x26 * (10x26)'
a3 = sigmoid (z3);
% z3, a3 = 5000x10


% 4) y 값의 벡터화는 mxn 형태로 만든다.
% ---------------------------------------
% y값을 벡터화 한다.
% y는 각 값에 대해서 하나의 컬럼벡터를 가져야 한다. 
% nxm 형태는 벡터 계산시 복잡하므로 mxn형태로 만든다.
% y_exp = zeros(5000, 10);		y_exp: 5000x10

% 아래와 같은 코드인데, eye() 함수를 사용하는 것이 좋다.
%y_exp = zeros(m, num_labels);
%for i = 1:m
%	y_exp(i, y(i)) = 1;
%end

eye_matrix = eye(num_labels);
y_exp = eye_matrix(y,:);


% 5) cost J(theta) 값을 구한다. J: 1x10
% ---------------------------------------
% 벡터 * 행렬은 .* 를 사용해서 요소 곱으로 해야 한다.
% sum() 함수는 컬럼기준으로 모든 요소의 합을 구한다.
J = 1/m * sum(-y_exp .* log(a3) - (1 - y_exp) .* log (1 - a3));
J = sum(J);		% unregularized cost (first answer)



% 6) regulization term을 구한다.
% ---------------------------------------
% bias 컬럼은 제외한다.
% sum()은 컬럼 단위로 적용된다.
sum_theta_square1 = sum(sum(Theta1(:, 2:end) .^ 2));
sum_theta_square2 = sum(sum(Theta2(:, 2:end) .^ 2));

reg_term = lambda / (2 * m) * (sum_theta_square1 + sum_theta_square2);

J = J + reg_term;		% second answer


% PART 2: 
% ================

% Delta 는 Theta와 dimension이 동일하다. 미리 초기화 한다.
Delta1 = zeros(size(Theta1, 1), size(Theta1, 2));
Delta2 = zeros(size(Theta2, 1), size(Theta2, 2));

% 모든 샘플들에 대해서 for-loop를 실행한다.
% X: 5000x401
for t = 1:m

	% step 1:
	% -------------------------------------------------
	a1 = X(t, :);			% a1 = 1x401
	z2 = a1 * Theta1';		% 1x401 x (25x401)'
	a2 = sigmoid(z2);
	% z2, a2: 1x25

	% Add ones to the a2 data matrix
	% a2 = [ones(size(a2, 1), 1) a2];
	a2 = [1 a2];
	z3 = a2 * Theta2'; 		% 1x26 x (10x26)'
	a3 = sigmoid (z3);
	% z3, a3 = 1x10


	% step 2:
	% -------------------------------------------------
	d3 = (a3 - y_exp(t, :));		% d3: 1x10

	% assignment reference의 논리연산자는 사용하지 않아도 된다. 아래 코드는 X 임.
	%[v, iv] = max(a3);
	%a3 = zeros(size(a3));
	%a3(iv) = 1;
	%d3 = !(a3 == y_exp(t, :));		% d3: 1x10


	% step3:
	% z2, d2: 25x1
	% Theta2: 10x26
	% Theta2' * d3' = 26x1
	% XXX: d2 계산 시 Theta2의 bias 항목이 제외되어야 한다.
	% -------------------------------------------------
	%d2 = Theta2' * d3' .* sigmoidGradient(z2);
	d2 = d3 * Theta2(:,2:end); 		% 1x10 * 10x25
	d2 = d2 .* sigmoidGradient(z2);

	
	% step 4:
	% -------------------------------------------------
	% Theta1: 25x401
	% Theta2: 10x26
	%
	% d2: 1x25, a1: 1x401
	% Delta1 = 25x401
	%
	% d3: 1x10, a2: 1x26
	% Delta2 = 10x26
	
	Delta1 = Delta1 + d2' * a1;
	Delta2 = Delta2 + d3' * a2;

end


% step 5:
% -------------------------------------------------
Theta1_grad = 1/m * Delta1;
Theta1_grad(:,2:end) = Theta1_grad(:, 2:end) + lambda / m * Theta1(:,2:end);	% add regularization term

Theta2_grad = 1/m * Delta2;
Theta2_grad(:,2:end) = Theta2_grad(:, 2:end) + lambda / m * Theta2(:,2:end);	% add regularization term





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
