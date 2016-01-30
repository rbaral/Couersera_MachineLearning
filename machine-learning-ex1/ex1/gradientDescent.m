function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
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
    %       of the cost function (computeCost) and gradient here.
    %
	x = X(:,2);%start from the second element, as the first one is just 1
    h = theta(1) + (theta(2)*x); % compute the value using our hypothesis function h(theta) = h(theta_0) + theta_1*x

	%now, update theta_0 using the Gradient descent formula
	%don't be confused here with the index: theta_0 = theta(1), because the array index starts from 1 in MATLAB
    theta_zero = theta(1) - alpha * (1/m) * sum(h-y);
	%now update the value of theta_1, using the gradient descent formula, again theta(2) is the second element and is for theta_1
	% also, here we use .* to get the element wise multiplication i.e. * is a vector or matrix multiplication but .* is an element wise multiplication
    theta_one  = theta(2) - alpha * (1/m) * sum((h - y) .* x);
	%now, simultaneously update the value of theta_0 and theta_1
    theta = [theta_zero; theta_one];
    % ============================================================

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
	%just print and see if the cost function is decreasing, if not, we might need to adjust the learning parameter (alpha)
	disp(min(J_history));

end

end
