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

best_err = 99999;
best_C = 0;
best_sigma = 0;
for C=[1]
  for sigma=[0.1 0.2 0.3]
	  model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    val_pred = svmPredict(model, Xval);
		val_err = mean(double(val_pred ~= yval));

		if best_err > val_err
		  best_err = val_err;
			best_C = C;
			best_sigma = sigma;
	  end
  end
end
% =========================================================================
C = best_C;
sigma = best_sigma;
disp("Best C");
disp(C);
disp("Best sigma");
disp(sigma);
end
