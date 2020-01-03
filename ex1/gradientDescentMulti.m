function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
%A = ones(num_iters,1);  % for plot of J_history

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %


	  h = X * theta;
    theta = theta - alpha / m * X' * (h - y);
    
    %theta = theta - alpha * (1/m) * sum((h-y)'*X);    
    %theta = theta - alpha / m * X' * (X * theta - y);

    % ============================================================

    % Save the cost J in every iteration     
    %A(iter) = iter;
    J_history(iter) = computeCost(X, y, theta);

end

    figure (1);
    plot(1:num_iters, J_history, '-b', 'LineWidth', 2);
    xlabel('Iteration Number'); % Set the x-axis label
    ylabel('Cost Value'); % Set the y-axis label
    legend('Cost Value');  

end
