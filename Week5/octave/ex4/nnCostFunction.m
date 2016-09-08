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

% X: (N, D) +1 for bias term
% Theta1: (H, D+1)
% Theta2: (C, H+1)
% y: (N, 1)
% Feed Forward
X = [ones(size(X)(1), 1), X]; % bias trick (N, D+1)
z2 = X * Theta1'; % (N, H)
pre_a2 = sigmoid(z2);

a2 = [ones(size(pre_a2), 1), pre_a2]; % (N, H+1)
z3 = a2 * Theta2'; % (N, C)
a3 = sigmoid(z3);

incorrect_score = 1 - a3;
log_ic_scores = log(1 - a3);
row_ic = sum(log_ic_scores, 2); %since it includes correct terms we have to subtract it
row_ic = row_ic - diag(log_ic_scores(:, y));
correct_scores = diag(a3(:, y)); % (N, 1)
log_c_scores = log(correct_scores);

%-1 * correct_scores - log(1 - incorrect_scores)
J = mean(-1 .* log_c_scores - row_ic);
reg_sum = sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2)); % regularization term
reg_term = lambda .* reg_sum ./ (2 * m); 
J += reg_term;

% -------------------------------------------------------------
% Theta2_grad should be (C, H+1)
% a3:= (N, C) a2:= (N, H+1), a3 - y := (N, C)
% Theta1_grad should be (H, D+1)
% Theta1_grad = Theta2(:, 2:end)' * (a3-y)' .* pre_a2 .* (1 - pre_a2) * X
y_matrix = zeros(m, num_labels);
idx = sub2ind(size(y_matrix), 1:size(y_matrix, 1), y');
y_matrix(idx) = 1;

Theta2_grad = (a3 - y_matrix)' * a2;
Theta1_grad = (a3 - y_matrix) * Theta2(:, 2:end) .* sigmoidGradient(z2);
Theta1_grad = Theta1_grad' * X;

Theta1_grad /= m;
Theta2_grad /= m;


% Regularization Terms
temp_Theta1 = Theta1;
temp_Theta1(:, 1) = 0;
temp_Theta2 = Theta2;
temp_Theta2(:, 1) = 0;
Theta1_grad += lambda * temp_Theta1/m;
Theta2_grad += lambda * temp_Theta2/m;
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];



end
