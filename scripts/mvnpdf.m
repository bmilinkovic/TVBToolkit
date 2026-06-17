function pdf = mvnpdf(x, mu, sigma)
%MVNPDF Minimal MATLAB-safe multivariate normal pdf.
%
% This local shim intentionally shadows the Octave-only implementation that is
% pulled in by ``addpath(genpath('/Users/borjan/code/matlab/elph'))``.
%
% Supported forms:
%   mvnpdf(x)
%   mvnpdf(x, mu)
%   mvnpdf(x, mu, sigma)
%
% Inputs:
%   x     [n x p] rows are observations
%   mu    scalar or [1 x p] mean
%   sigma scalar or [p x p] covariance

if nargin < 2 || isempty(mu)
    mu = 0;
end
if nargin < 3 || isempty(sigma)
    sigma = 1;
end

x = double(x);
if ~ismatrix(x)
    error('mvnpdf: first input must be a 2D matrix');
end

[n, p] = size(x);

if isscalar(mu)
    mu = repmat(double(mu), 1, p);
else
    mu = double(mu(:).');
end
if numel(mu) ~= p
    error('mvnpdf: mean dimension does not match observation dimension');
end
mu = repmat(mu, n, 1);

if isscalar(sigma)
    sigma = eye(p) .* double(sigma);
else
    sigma = double(sigma);
end
if ~ismatrix(sigma) || size(sigma, 1) ~= p || size(sigma, 2) ~= p
    error('mvnpdf: covariance must be scalar or square [p x p]');
end

sigma = 0.5 .* (sigma + sigma.');
eps_diag = max(1e-12, 1e-9 * max(abs(diag(sigma))));
sigma = sigma + eye(p) .* eps_diag;

[R, chol_flag] = chol(sigma);
if chol_flag ~= 0
    [V, D] = eig(sigma);
    d = diag(D);
    d(d < 1e-12) = 1e-12;
    sigma = V * diag(d) * V';
    sigma = 0.5 .* (sigma + sigma.');
    [R, chol_flag] = chol(sigma);
    if chol_flag ~= 0
        error('mvnpdf: covariance matrix is not positive definite');
    end
end

xc = x - mu;
quad = sum((xc / R) .^ 2, 2);
norm_const = (2 * pi) ^ (-p / 2) / prod(diag(R));
pdf = norm_const .* exp(-0.5 .* quad);
