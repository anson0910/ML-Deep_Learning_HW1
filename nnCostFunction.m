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

% Reshape nn_params back into the parameters, the weight matrices
% for our 2 layer neural network
W1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size)), ...
                 hidden_layer_size, input_layer_size);           %  128 x 69
             
b1 = reshape(nn_params(1 + hidden_layer_size * (input_layer_size)   :     hidden_layer_size * (input_layer_size) + hidden_layer_size ), ...
                 hidden_layer_size, 1 );           %  128 x 1
             
W2 = reshape(nn_params( 1 +   hidden_layer_size * (input_layer_size)  +  hidden_layer_size  :  hidden_layer_size * (input_layer_size)  +  hidden_layer_size + num_labels * hidden_layer_size ), ...
                 num_labels, hidden_layer_size );                   %  48 x 128
             
b2 = reshape(nn_params( 1 + hidden_layer_size * (input_layer_size)  +  hidden_layer_size + num_labels * hidden_layer_size  :  end ), ...
                 num_labels, 1 );                   %  48 x 1

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
W1_grad = zeros(size(W1));
b1_grad = zeros(size(b1));
W2_grad = zeros(size(W2));
b2_grad = zeros(size(b2));


% Part 1: Feedforward the neural network and return the cost in the
%         variable J. 
%{
%}

for cur_example = 1:m
        
        y_cur_example = y( : , cur_example);     %48 x 1
        X_cur_example = X( : , cur_example) ;    % 69 x 1
        z_1 = W1 * X_cur_example + b1;   % 128 x 1
        a_1 = sigmoid(z_1) ;                     % 128 x 1
        z_2 = W2 * a_1 + b2 ;                   % 48 x 1
        a_2 = sigmoid(z_2) ;                      % 48 x 1  (output)
        
       % cur_example_cost = - expanded_y( : , cur_example) .* log(a_3)  -  (1 - expanded_y( : , cur_example)) .*  log(1 - a_3) ;
       cur_example_cost = (a_2  -  y_cur_example) .^ 2;
      
        delta_2 = sigmoidGradient( z_2) * 2 .* (a_2 - y_cur_example) ;
        delta_1 = sigmoidGradient( z_1) .*  ( W2'  * delta_2);
        
        W2_grad = delta_2 * a_1' ;    % 48 x 128
        b2_grad = delta_2 ;                 % 48 x 1
        W1_grad = delta_1 * X' ;        % 128 x 69
        b1_grad = delta_1 ;                    % 128 x 1
       
        J = J + sum(cur_example_cost);
end

J = J/m  ;
%+  lambda/2/m * ( sum(sum(W1( : , 2:end ) .^2 )) + sum(sum(W2( : , 2:end ) .^2 ))  );
%{
%}

%  Implement regularization with the cost function and gradients.
% Compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

regterm_W1 = lambda/m * W1_temp;
regterm_W2 = lambda/m * W2_temp;

%W1_grad = W1_grad + regterm_W1;
%W2_grad = W2_grad + regterm_W2;

% -------------------------------------------------------------

% =========================================================================
%}
% Unroll gradients
grad = [W1_grad(:) ; b1_grad(:) ; W2_grad(:) ; b2_grad(:)];


end
