function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

mu = mean(X);%mean(X) computes the mean of each column separately and returns the vector of means for each column
sigma = std(X); %computes the standard deviation of each column

%replicate the mean array into vector of size length(X) (ie length(X)*1)
numerator = (X - repmat(mu, length(X), 1));
%replicate the std array into vector of size length(X) (ie length(X)*1)
denominator = repmat(sigma, length(X), 1);
%now apply the normalization formula. Here we use ./ to have element wise
%division
X_norm = (numerator) ./ denominator;

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       









% ============================================================

end
