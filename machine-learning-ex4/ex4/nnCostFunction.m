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
J_unreg=0;
y_label= zeros(m,num_labels);
for i= 1:m
    y_label(i,y(i))=1 ;
end
%{
a1= X;
a1_one= [ones(m,1) X];
a2= sigmoid(a1_one*Theta1');
a2_one= [ones(m,1) a2];
a3= sigmoid(a2_one*Theta2');
a3_one=[ones(m,1) a3];

h= a3;

for i= 1:size(Theta1,1)
    for j=2:size(Theta1,2)
        temp1(i,j-1)=Theta1(i,j);
    end
end

for i= 1:size(Theta2,1)
    for j=2:size(Theta2,2)
        temp2(i,j-1)=Theta2(i,j);
    end
end

for i=1:m
    for k= 1:num_labels
        J_unreg= J_unreg + 1/m*(-y_label(i,k)*log(h(i,k))-(1-y_label(i,k))*log(1-h(i,k)));
    end
end
J= J_unreg + lambda/(2*m)*(sum(sum(temp1.^2))+sum(sum(temp2.^2)));

z2= a1_one*Theta1'; z3= a2_one*Theta2';
%{
[M,a3_h]= max(a3,[],2);
dif= (a3_h==y);
delta3= zeros(m,num_labels);
for i= 1:m
    if dif(i)==1
        delta3(i,dif(i))=1;
    end
end
%}
t1= Theta1; t2= Theta2;
for i=1:size(Theta1,1)
    t1(i,1)=0; 
end
for i=1:size(Theta2,1)
    t2(i,1)=0; 
end
delta3= a3-y_label;
delta2= delta3*temp2.*sigmoidGradient(z2);
Theta1_grad = Theta1_grad + 1/m*(delta2'*a1_one) + lambda/m*t1;
Theta2_grad = Theta2_grad + 1/m*(delta3'*a2_one) + lambda/m*t2;

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
%}
w1= Theta1;w2=Theta2;x0=X;w1_grad= zeros(size(w1));w2_grad= zeros(size(w2));
a0= [ones(size(x0,1),1) X];
z1= a0*w1';
a1= sigmoid(z1);
x1= [ones(size(x0,1),1) a1];
x2= sigmoid(x1*w2');
h=x2;

J_unreg=1/m*( -y_label.*log(h)-(1-y_label).*log(1-h) );
J_unreg= sum(sum(J_unreg));
%1/m*(-y_label(i,k)*log(h(i,k))-(1-y_label(i,k))*log(1-h(i,k)));
J= J_unreg + lambda/(2*m)* ( sum(sum(w1(:,2:end).^2)) + sum(sum(w2(:,2:end).^2)) );
%J= J_unreg + lambda/(2*m)*(sum(sum(temp1.^2))+sum(sum(temp2.^2)));

delta2= x2- y_label;
delta1= delta2*w2.*sigmoidGradient([ones(size(z1,1),1) z1]);
%delta3= a3-y_label;
%delta2= delta3*temp2.*sigmoidGradient(z2);
t1=w1;t2=w2;
for i=1:size(w1,1)
    t1(i,1)=0;
end
for i=1:size(w2,1)
    t2(i,1)=0;
end
w1_grad= w1_grad + 1/m*(a0'*delta1(:,2:end))' + lambda/m*t1;
w2_grad= w2_grad + 1/m*(x1'*delta2)' + lambda/m*t2;

grad= [w1_grad(:) ; w2_grad(:) ];

%Theta1_grad = Theta1_grad + 1/m*(delta2'*a1_one) + lambda/m*t1;
%Theta2_grad = Theta2_grad + 1/m*(delta3'*a2_one) + lambda/m*t2;

end
