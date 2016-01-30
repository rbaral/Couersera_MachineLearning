function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    % Calculate the hypothesis function h

    h = X * theta;

    % Calculate the next iteration of theta by learning rate alpha.  
    % Equivalently, we can also do the followings:
    % nextTheta0 = theta(1) - (alpha * (1 / m) * sum(h - y));%this is for first
    % theta
    % nextTheta1 = theta(2) - (alpha * (1 / m) * sum((h - y) .* X(:, 2))); %
    % for second theta
    % theta = [nextTheta0; nextTheta1]; % simultaneous update of theta
    %
    % The approach fully utilizes matrix operations though, so it may generalize to
    % multivariate linear regression.

    theta = theta - alpha * (1 / m) * (X' * (h - y));


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
