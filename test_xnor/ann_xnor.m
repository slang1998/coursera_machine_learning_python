% sigmoid.m 함수 구현
% sigmoidGradient.m 함수 구현
% cost 함수 nnCostFunction.m 구현

% xnor를 구현한다.
% 2개의 입력, 2개의 히든 유닛, 4개의 출력을 갖는다.
% 출력: 1 => [1 0 0 0]', 2 => [0 1 0 0]', 3 => [0 0 1 0]', 4 => [0 0 0 1]'

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
%  전체 3개의 레이어를 가지며, 그 중 1개가 hidden layer이다.
input_layer_size  = 2;  	% 2 Input signal
hidden_layer_size = 2;   	% 2 hidden units
num_labels = 4;          	% 4 labels, from 1 to 4
                          


%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
% 트레이닝 데이터를 로딩하고 그래프로 그려본다.

X = load('data_X.mat');
y = load('data_y.mat');
m = size(X, 1);

% 입력값 X = 10x2 
% 출력값 y = 10x1
%X = [0 0; 0 1; 1 0; 1 1; 0 0; 0 1; 1 0; 1 1;0 0; 0 1];
%y = [1; 2; 3; 4; 1; 2; 3; 4; 1; 2];
%m = size(X, 1);


%% ================ Part 3: Compute Cost (Feedforward) ================
% 인공신경망에서는, 비용만 반환하는 인공 신경망의 feedforward 부분을 먼저
% 구현해야 한다. 이 계산은 nnCostFunction.m 에서 이루어지며, 올바르게 
% 구현되었는지를 확인하기 위해서 디버깅 파라미터로 동일한 비용이 나오는지 
% 확인해야 한다.
%
% 먼저 처음에는 regularization 없는 freedforward cost를 구현하기 권장한다. 
% 그래야 디버깅하기 쉬울 것이다.
%

% NOTE: 1개의 hidden layer를 가지고 있으므로 2개의 parameter Theta값이 필요한다.
% 파라미터 풀어헤치기
Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

nn_params = [Theta1(:) ; Theta2(:)];


% Weight regularization parameter (we set this to 0 here).
lambda = 0;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Cost at parameters (random initialized): %f \n '], J);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% =============== Part 4: Implement Regularization ===============
%  일단, cost function 구현이 정확하면, 비용과 함께 regularization을
%  구현한다.
%

fprintf('\nChecking Cost Function (w/ Regularization) ... \n')

% Weight regularization parameter (we set this to 1 here).
lambda = 1;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Cost at parameters : %f \n'], J);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 5: Sigmoid Gradient  ================
%  인공신경망을 구현하기 전에, sigmoid 함수에 대한 gradient를 구현
%  해야 한다. 이것은 sigmoidGradient.m 파일에 있고, 크게 변경될 일이 
%  없어서 기존 것을 그대로 사용한다.
%


%% ================ Part 6: Initializing Pameters ================
%  여기서는 xnor 로직을 구현하기 위해서(분류하기 위해서) 2개의 레이어를
%  구현한다. 이때 가중치(theata 값)는 랜덤으로 구하도록 한다.
%  기존의 randInitializeWeights() 함수를 사용한다.
%  (randInitializeWeights.m)
%
% NOTE: Theta 값 설정은 hidden layer 갯수에 따라 달라진다.

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];



%% =============== Part 7: Implement Backpropagation ===============
%  일단, 비용함수가 일치하면(올바르다면) 인공신경망을 위한 backpropagation
%  알고리즘을 구현한다. nnCostFunction.m 파일에서 파라미터의 편미분값을 
%  반환하도록 구현한다.
%  checkNNGradients 는 debugInitializeWeights() 통해서 초기 Theta 값들을
%  생성하고, nnCostFunction의 결과값과 computeNumericalGradient의 결과값을
%  비교해 보는 프로그램이다. 2개의 결과값은 매우 비슷해야 한다.
%
fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
checkNNGradients;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;



%% =============== Part 8: Implement Regularization ===============
%  일단, backpropagation 구현이 올바르면, cost와 gradient를 가진 
%  regularization을 구현한다.
%

fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')

%  Check gradients by running checkNNGradients
lambda = 3;
checkNNGradients(lambda);

% Also output the costFunction debugging values
debug_J  = nnCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);

fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = 10): %f ' ...
         '\n(this value should be about 0.576051)\n\n'], debug_J);

fprintf('Program paused. Press enter to continue.\n');
pause;



%% =================== Part 8: Training NN ===================
%  이제 인공신경망을 훈련시키기 위한 모든 코드를 구현했다. 
%  인공신경망을 훈련시키기 위해서 fmincg를 사용할 것인데, 이 함수는 
%  fminunc와 유사하게 동작하는 함수이다. 이들 advanced optimizers 함수들은
%  cost functions들을 효율적으로 train 시킬 수 있다.
%

fprintf('\nTraining Neural Network... \n')

%  반복횟수 MaxIter값을 조정하면서 training에 얼마나 도움이 되는지 확인해보자.
options = optimset('MaxIter', 50);

%  lambda 값을 서로 다르게 해보고 테스트 해보자.
lambda = 1;

% 최소화 되어야 하는 cost function에 대한 "short hand"를 생성한다.
costFunction = @(p) nnCostFunction(p, ...
                    input_layer_size, ...
                    hidden_layer_size, ...
                    num_labels, X, y, lambda);


% 이제, costFunction은 단 하나의 인자(인공신경망 파라미터)만 취하는 함수가 된다.
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);


% nn_params 에서 Theta1, Theta2 값을 얻는다.
% 훈련된 인공신경망의 theta 값이 된다.
% 
% NOTE: Theta 값 설정은 hidden layer 갯수에 따라 달라진다.
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');


%% ================= Part 9: Visualize Weights =================
%  이제 hidden layer를 표시해서 인공신경망이 얻은 것을 "visualize"
%  할 수 있게 된다. 이것은 데이터에서 캡쳐된 feature가 된다.
% 
%  NOTE: 여기서는 그래프를 확인하지 않는다.

%fprintf('\nVisualizing Neural Network... \n')

%displayData(Theta1(:, 2:end));

%fprintf('\nProgram paused. Press enter to continue.\n');
%pause;

%% ================= Part 10: Implement Predict =================
%  인공신경망을 훈련시킨 후에, labels를 예측할 수 있게 된다.
%  "predict" 함수를 구현해서 training set의 라벨을 예측하는데 사용된다.
%  이것은 training set 의 정확도 accuracy를 계산하게 해준다.
% 
%  NOTE: predict() 함수는 크게 달라지는 부분이 없으므로 기존 것을 그대로 
%  사용한다. 다만, hidden layer 갯수가 달라지면, predict() 함수 구현도
%  달라야 한다.
%
% ex) predict(Theta1, Theta2, [0 0;1 0])
%

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);





