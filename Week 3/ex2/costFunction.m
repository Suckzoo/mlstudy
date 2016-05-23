function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

for i = 1:m
    hValue = sigmoid(X(i,:) * theta);
    cost = - y(i) * log(hValue) - (1 - y(i)) * log(1 - hValue);
    J = J + cost;
    for j = 1:n
        grad(j) = grad(j) + (hValue - y(i)) * X(i, j);
    end
end

J = J / m;
grad = grad / m;

% =============================================================

end
