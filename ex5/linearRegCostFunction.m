function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

J = sum((X * theta - y) .^ 2) / (2*m);
J = J + lambda / (2*m) * sum(theta(2:n) .^ 2);

grad = (X' * (X * theta - y)) / m;
grad = grad + lambda ./ m .* theta .* [0; ones(n-1,1)];

% =========================================================================

grad = grad(:);

end
