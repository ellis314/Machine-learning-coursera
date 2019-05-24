function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
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
C_num=[0.01 0.03 0.1 0.3 1 3 10 30];
sigma_num=[0.01 0.03 0.1 0.3 1 3 10 30];
for i= 1:size(C_num,2)
    for j=1:size(sigma_num,2)
        model= svmTrain(X, y, C_num(i), @(x1, x2) gaussianKernel(x1, x2, sigma_num(j)));
        p= svmPredict(model,Xval);
        error(i,j)= mean(double(p~=yval));
    end
end


[M_i,I_i]=min(error);
[M_j,I_j]=min(M_i);
i= I_i(I_j);
j= I_j; 
C=C_num(i);
sigma=sigma_num(j);




% =========================================================================

end
