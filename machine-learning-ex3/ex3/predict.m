function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Theta1: 25x401
% Theta2: 10x26
% X: 5000x400 => 5000x401

% Input layer: 400 units
% Second layer: 25 units

% Add ones to the X data matrix
X = [ones(m, 1) X];
z = Theta1 * X'; 
a2 = sigmoid (z);
% z, a2: 25x5000

a2 = a2';
% a2: 5000x25

% Add ones to the a2 data matrix
a2 = [ones(size(a2, 1), 1) a2];
z = Theta2 * a2'; 
a3 = sigmoid (z);

% �� ������ �÷����� �����Ǿ� �ִ�. �÷����� �ִ밪�� ���ϰ� �� ��ġ���� ��ȸ�Ѵ�.
% z, a3 = 10x5000

% �÷� �������� �ִ밪�� ���� ���ͷ� �޴´�.
[x, ix] = max(a3, [], 1);


p = ix;

% �÷� ���ͷ� ��ȯ
p = p(:);

% =========================================================================


end
