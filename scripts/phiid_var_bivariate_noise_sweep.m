function phiid_var_bivariate_noise_sweep(config_path, output_path, use_parallel, n_workers)
%PHIID_VAR_BIVARIATE_NOISE_SWEEP Simulate a bivariate VAR(1) PhiID noise sweep.
%
% This runner simulates a latent bivariate VAR(1),
%   z_t = A z_{t-1} + sigma_eps * eps_t
% and then adds observation noise
%   x_t = z_t + sigma_obs * eta_t
% before computing PhiIDFull(x, tau, measure).
%
% The sweep is over observation-noise standard deviation because PhiIDFull
% internally standardizes each variable; a pure innovation-amplitude sweep
% would largely rescale the process without changing its correlation
% structure, making it a weak manipulation for this purpose.

if nargin < 3 || isempty(use_parallel)
    use_parallel = false;
end
if nargin < 4 || isempty(n_workers)
    n_workers = 0;
end

cfg = load(config_path);
noise_levels = row_vector_double(cfg.noise_levels);
measures = matlab_string_list(cfg.measures);
n_noise = numel(noise_levels);
n_measures = numel(measures);
n_replicates = scalar_int(cfg.n_replicates);
n_timepoints = scalar_int(cfg.n_timepoints);
burnin = scalar_int(cfg.burnin);
tau = scalar_int(cfg.tau);
self_coef = scalar_double(cfg.self_coef);
cross_coef = scalar_double(cfg.cross_coef);
innovation_sd = scalar_double(cfg.innovation_sd);
common_noise_fraction = scalar_double(cfg.common_noise_fraction);
base_seed = scalar_int(cfg.base_seed);

A = [self_coef, cross_coef; cross_coef, self_coef];
eigvals = eig(A);
if any(abs(eigvals) >= 1.0)
    error('VAR coefficient matrix is not stable: max abs eigenvalue = %.4f', max(abs(eigvals)));
end

job_seeds = base_seed + (1:(n_noise * n_replicates))';
sts_linear = nan(n_noise * n_replicates * n_measures, 1);
rtr_linear = nan(n_noise * n_replicates * n_measures, 1);
status_linear = zeros(n_noise * n_replicates * n_measures, 1, 'int8');

if use_parallel
    pool = gcp('nocreate');
    if isempty(pool)
        if n_workers > 0
            parpool('local', n_workers);
        else
            parpool('local');
        end
        pool = gcp('nocreate');
    end
    if ~isempty(pool)
        fprintf('Connected to parallel pool with %d workers.\n', pool.NumWorkers);
    end
end

fprintf('Running bivariate VAR PhiID noise sweep | n_noise=%d | n_replicates=%d | n_measures=%d\n', ...
    n_noise, n_replicates, n_measures);

measure_index = zeros(n_noise * n_replicates * n_measures, 3);
cursor = 0;
for measure_idx = 1:n_measures
    for noise_idx = 1:n_noise
        for rep_idx = 1:n_replicates
            cursor = cursor + 1;
            measure_index(cursor, :) = [noise_idx, rep_idx, measure_idx];
        end
    end
end

if use_parallel
    parfor job_idx = 1:size(measure_index, 1)
        [sts_val, rtr_val, status_code] = run_single_job( ...
            measure_index(job_idx, :), noise_levels, measures, job_seeds, ...
            A, innovation_sd, common_noise_fraction, n_timepoints, burnin, tau, n_replicates);
        sts_linear(job_idx) = sts_val;
        rtr_linear(job_idx) = rtr_val;
        status_linear(job_idx) = status_code;
    end
else
    for job_idx = 1:size(measure_index, 1)
        [sts_val, rtr_val, status_code] = run_single_job( ...
            measure_index(job_idx, :), noise_levels, measures, job_seeds, ...
            A, innovation_sd, common_noise_fraction, n_timepoints, burnin, tau, n_replicates);
        sts_linear(job_idx) = sts_val;
        rtr_linear(job_idx) = rtr_val;
        status_linear(job_idx) = status_code;
    end
end

sts_values = permute(reshape(sts_linear, [n_replicates, n_noise, n_measures]), [2, 1, 3]);
rtr_values = permute(reshape(rtr_linear, [n_replicates, n_noise, n_measures]), [2, 1, 3]);
status_codes = permute(reshape(status_linear, [n_replicates, n_noise, n_measures]), [2, 1, 3]);

job_seed_matrix = reshape(job_seeds, [n_replicates, n_noise])';
save(output_path, ...
    'noise_levels', 'measures', 'sts_values', 'rtr_values', 'status_codes', ...
    'job_seed_matrix', 'self_coef', 'cross_coef', 'innovation_sd', ...
    'common_noise_fraction', 'n_timepoints', 'burnin', 'tau', 'base_seed', ...
    '-v7');

end


function [sts_val, rtr_val, status_code] = run_single_job(job_triplet, noise_levels, measures, job_seeds, A, innovation_sd, common_noise_fraction, n_timepoints, burnin, tau, n_replicates)
noise_idx = job_triplet(1);
rep_idx = job_triplet(2);
measure_idx = job_triplet(3);
noise_sd = noise_levels(noise_idx);
measure = measures{measure_idx};

seed_idx = (noise_idx - 1) * n_replicates + rep_idx;
rng(job_seeds(seed_idx), 'twister');

observed = simulate_bivariate_var_observed(A, innovation_sd, noise_sd, common_noise_fraction, n_timepoints, burnin);

try
    atoms = PhiIDFull(observed, tau, measure);
    sts_val = scalar_double(atoms.sts);
    rtr_val = scalar_double(atoms.rtr);
    status_code = int8(1);
catch ME
    warning('PhiID job failed (measure=%s, noise_idx=%d, rep_idx=%d): %s', ...
        measure, noise_idx, rep_idx, ME.message);
    sts_val = nan;
    rtr_val = nan;
    status_code = int8(-1);
end
end


function observed = simulate_bivariate_var_observed(A, innovation_sd, observation_sd, common_noise_fraction, n_timepoints, burnin)
total_steps = n_timepoints + burnin;
latent = zeros(2, total_steps);

for t = 2:total_steps
    latent(:, t) = A * latent(:, t - 1) + innovation_sd * randn(2, 1);
end

latent = latent(:, burnin + 1:end);

private_scale = sqrt(max(0.0, 1.0 - common_noise_fraction));
common_scale = sqrt(max(0.0, common_noise_fraction));
common_obs = common_scale * repmat(randn(1, n_timepoints), 2, 1);
private_obs = private_scale * randn(2, n_timepoints);
observed = latent + observation_sd * (common_obs + private_obs);
end


function values = row_vector_double(x)
values = double(x(:).');
end


function value = scalar_double(x)
value = double(x(1));
end


function value = scalar_int(x)
value = round(double(x(1)));
end


function values = matlab_string_list(raw)
if ischar(raw) || isstring(raw)
    values = cellstr(raw);
    return
end

if ~iscell(raw)
    values = cellstr(string(raw));
    return
end

values = cell(size(raw));
for idx = 1:numel(raw)
    item = raw{idx};
    if isstring(item)
        values{idx} = char(item);
    elseif ischar(item)
        values{idx} = item;
    else
        values{idx} = char(string(item));
    end
end
values = values(:).';
end
