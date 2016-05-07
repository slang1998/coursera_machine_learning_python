function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% 입력값 X에서 첫번째 hidden layer에 대한 출력값을 구한다. (h1)
% h1을 입력값으로 사용해서 최종 output layer에 대한 출력값을 구한다. (h2)
% 여기서 가장 큰 값을 선택한다.
h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');

% by skkong
size(h2)

h2(1,:)

[dummy, p] = max(h2, [], 2);

% =========================================================================


end
