function X = mvnrnd(mu, Sigma, n)
%MVNRND Minimal MATLAB-safe multivariate normal sampler.
%   X = MVNRND(MU, SIGMA, N) returns N samples from N(MU, SIGMA).
%
% This shim exists to avoid pulling the legacy Octave helper from the
% upstream elph toolbox, which can fail under MATLAB. It only implements the
% usage pattern needed by the PhiID CCS path.

if nargin < 2
    error('mvnrnd requires at least mu and Sigma.');
end
if nargin < 3 || isempty(n)
    n = 1;
end

mu = double(mu(:).');
Sigma = double(Sigma);
n = double(n);

if ~ismatrix(Sigma) || size(Sigma, 1) ~= size(Sigma, 2)
    error('Sigma must be a square covariance matrix.');
end
if size(Sigma, 1) ~= numel(mu)
    error('mu length must match Sigma dimensions.');
end
if n < 1 || floor(n) ~= n
    error('n must be a positive integer.');
end

% Symmetrize for numerical stability.
Sigma = (Sigma + Sigma.') ./ 2.0;

[R, p] = chol(Sigma, 'lower');
if p ~= 0
    jitter = 1e-10 * eye(size(Sigma));
    [R, p] = chol(Sigma + jitter, 'lower');
    if p ~= 0
        error('Sigma must be positive semidefinite.');
    end
end

Z = randn(n, numel(mu));
X = Z * R.' + repmat(mu, n, 1);

end
