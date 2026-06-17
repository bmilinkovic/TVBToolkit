function phiid_empirical_bold_local_sts_rtr_aal90(input_dir, output_dir, redundancy, use_parallel, n_workers)
%PHIID_EMPIRICAL_BOLD_LOCAL_STS_RTR_AAL90 Save local STS/RTR edge time series.
%
% For each subject, this computes [A, L] = PhiIDFull([roi_i; roi_j], 1, redundancy)
% on every unordered ROI pair and saves the local atoms L.sts and L.rtr over time.
%
% Output file per subject:
%   <subject_stub>__local_sts_rtr_<redundancy>.mat

if nargin < 1 || strlength(string(input_dir)) == 0
    error('input_dir is required.');
end
if nargin < 2 || strlength(string(output_dir)) == 0
    error('output_dir is required.');
end
if nargin < 3 || strlength(string(redundancy)) == 0
    redundancy = 'mmi';
end
if nargin < 4 || isempty(use_parallel)
    use_parallel = true;
end
if nargin < 5 || isempty(n_workers)
    n_workers = 0;
end

input_dir = char(string(input_dir));
output_dir = char(string(output_dir));
redundancy = char(string(redundancy));
use_parallel = logical(use_parallel);
n_workers = double(n_workers);

if ~exist(input_dir, 'dir')
    error('Input directory does not exist: %s', input_dir);
end
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

files = dir(fullfile(input_dir, '*.mat'));
if isempty(files)
    error('No .mat input files found in: %s', input_dir);
end
file_names = {files.name};

run_parallel = false;
if use_parallel
    try
        has_pct = license('test', 'Distrib_Computing_Toolbox');
    catch
        has_pct = false;
    end
    if has_pct
        pool = gcp('nocreate');
        if isempty(pool)
            if n_workers > 0
                parpool('local', n_workers);
            else
                parpool('local');
            end
            pool = gcp('nocreate');
        end
        run_parallel = ~isempty(pool);
    end
end

fprintf('Local PhiID redundancy=%s | subject_parallel=%d | requested_workers=%d\n', redundancy, run_parallel, n_workers);

if run_parallel
    parfor file_idx = 1:length(file_names)
        process_local_subject(file_names{file_idx}, file_idx, length(file_names), input_dir, output_dir, redundancy);
    end
else
    for file_idx = 1:length(file_names)
        process_local_subject(file_names{file_idx}, file_idx, length(file_names), input_dir, output_dir, redundancy);
    end
end

end


function process_local_subject(file_name, file_idx, n_files, input_dir, output_dir, redundancy)

subject_stub = erase(file_name, '.mat');
output_path = fullfile(output_dir, [subject_stub '__local_sts_rtr_' redundancy '.mat']);
if exist(output_path, 'file')
    fprintf('Local PhiID %s (%d/%d): %s [skip existing]\n', redundancy, file_idx, n_files, file_name);
    return
end

data = load(fullfile(input_dir, file_name));
if isfield(data, 'time_series')
    time_series = double(data.time_series);
elseif isfield(data, 'bold_timeseries')
    time_series = double(data.bold_timeseries);
elseif isfield(data, 'source_ts')
    time_series = double(data.source_ts);
else
    error('No supported timeseries variable found in %s', file_name);
end

if size(time_series, 1) ~= 90 && size(time_series, 2) == 90
    time_series = time_series.';
end
if size(time_series, 1) ~= 90
    error('Expected 90 regions in %s, got size %dx%d', file_name, size(time_series, 1), size(time_series, 2));
end

n_regions = size(time_series, 1);
n_timepoints = size(time_series, 2);
local_len = max(n_timepoints - 1, 0);
row_std = std(time_series, 0, 2);
degenerate_rows = find(~isfinite(row_std) | row_std <= 1e-12);
[edge_i, edge_j] = find(triu(true(n_regions), 1));
n_edges = length(edge_i);

sts_edges = zeros(local_len, n_edges, 'single');
rtr_edges = zeros(local_len, n_edges, 'single');
skipped_edges = false(n_edges, 1);

fprintf('Local PhiID %s (%d/%d): %s\n', redundancy, file_idx, n_files, file_name);
if ~isempty(degenerate_rows)
    fprintf('  Degenerate ROI rows for %s: %s\n', subject_stub, mat2str(degenerate_rows));
end

for edge_idx = 1:n_edges
    i = edge_i(edge_idx);
    j = edge_j(edge_idx);

    if ~isfinite(row_std(i)) || ~isfinite(row_std(j)) || row_std(i) <= 1e-12 || row_std(j) <= 1e-12
        skipped_edges(edge_idx) = true;
        continue
    end

    pair_ts = [time_series(i, :); time_series(j, :)];
    if any(~isfinite(pair_ts), 'all')
        skipped_edges(edge_idx) = true;
        continue
    end

    try
        [~, L] = PhiIDFull(pair_ts, 1, redundancy);
    catch
        skipped_edges(edge_idx) = true;
        continue
    end

    if ~isstruct(L) || ~isfield(L, 'sts') || ~isfield(L, 'rtr')
        skipped_edges(edge_idx) = true;
        continue
    end

    sts_vec = local_atom_column(L.sts, local_len);
    rtr_vec = local_atom_column(L.rtr, local_len);
    sts_edges(:, edge_idx) = sts_vec;
    rtr_edges(:, edge_idx) = rtr_vec;
end

subject_meta = struct();
subject_meta.subject_stub = subject_stub;
subject_meta.redundancy = redundancy;
subject_meta.local_len = local_len;
subject_meta.n_edges = n_edges;
subject_meta.n_regions = n_regions;
subject_meta.valid_edge_count = sum(~skipped_edges);
subject_meta.degenerate_rows = degenerate_rows;
if isfield(data, 'subject_id'), subject_meta.subject_id = data.subject_id; end
if isfield(data, 'cohort'), subject_meta.cohort = data.cohort; end
if isfield(data, 'stage'), subject_meta.stage = data.stage; end
if isfield(data, 'sedation'), subject_meta.sedation = data.sedation; end
if isfield(data, 'source_fc_file'), subject_meta.source_fc_file = data.source_fc_file; end
if isfield(data, 'source_sc_file'), subject_meta.source_sc_file = data.source_sc_file; end
if isfield(data, 'source_subject_index'), subject_meta.source_subject_index = data.source_subject_index; end
if isfield(data, 'source_subject_label'), subject_meta.source_subject_label = data.source_subject_label; end
if isfield(data, 'tr_seconds'), subject_meta.tr_seconds = data.tr_seconds; end
if isfield(data, 'roi_labels'), subject_meta.roi_labels = data.roi_labels; end

save(output_path, 'sts_edges', 'rtr_edges', 'edge_i', 'edge_j', 'skipped_edges', 'subject_meta', '-v7.3');

end


function col = local_atom_column(x, local_len)

col = zeros(local_len, 1, 'single');
if isempty(x) || local_len == 0
    return
end

vals = single(x(:));
n_copy = min(local_len, length(vals));
col(1:n_copy) = vals(1:n_copy);
col(~isfinite(col)) = 0;

end
