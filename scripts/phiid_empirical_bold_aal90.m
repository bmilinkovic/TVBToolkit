function phiid_empirical_bold_aal90(input_dir, output_dir, redundancy, use_parallel, n_workers)
%PHIID_EMPIRICAL_BOLD_AAL90 Run pairwise PhiID on empirical AAL90 BOLD.
%
% This mirrors the legacy EEG workflow traced from:
% - /Users/borjan/code/python/TVBEmergence/test/matlab/emergence_measures.m
%
% Inputs are exported from Python as .mat files containing:
% - time_series: [regions x time]
% - subject_id / cohort / stage / sedation metadata (optional)
%
% Outputs are saved one atom per file, following the legacy style but with a
% double-underscore subject delimiter:
%   <subject_stub>__sts_mat_idep_xtb.mat
%
% Example:
%   phiid_empirical_bold_aal90('results/phiid_empirical_bold/inputs', ...
%       'results/phiid_empirical_bold/phiid/mmi', 'mmi', true, 8)

if nargin < 1 || strlength(string(input_dir)) == 0
    error('input_dir is required.');
end
if nargin < 2 || strlength(string(output_dir)) == 0
    error('output_dir is required.');
end
if nargin < 3 || strlength(string(redundancy)) == 0
    redundancy = 'idep_xtb';
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

atom_stems = {
    'sts_mat', 'rtr_mat', 'rtx_mat', 'rty_mat', 'rts_mat', ...
    'xtr_mat', 'xtx_mat', 'xty_mat', 'xts_mat', ...
    'ytr_mat', 'ytx_mat', 'yty_mat', 'yts_mat', ...
    'str_mat', 'stx_mat', 'sty_mat', ...
    'sr_gradient'
};

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

fprintf('PhiID redundancy=%s | parallel=%d | requested_workers=%d\n', redundancy, run_parallel, n_workers);

if run_parallel
    parfor file_idx = 1:length(file_names)
        process_subject_file(file_names{file_idx}, file_idx, length(file_names), input_dir, output_dir, redundancy, atom_stems);
    end
else
    for file_idx = 1:length(file_names)
        process_subject_file(file_names{file_idx}, file_idx, length(file_names), input_dir, output_dir, redundancy, atom_stems);
    end
end

end


function tf = subject_outputs_complete(output_dir, subject_stub, redundancy, atom_stems)

tf = true;
for idx = 1:length(atom_stems)
    stem = atom_stems{idx};
    expected = fullfile(output_dir, [subject_stub '__' stem '_' redundancy '.mat']);
    if ~exist(expected, 'file')
        tf = false;
        return
    end
end

end


function process_subject_file(file_name, file_idx, n_files, input_dir, output_dir, redundancy, atom_stems)

file_path = fullfile(input_dir, file_name);
subject_stub = erase(file_name, '.mat');
if subject_outputs_complete(output_dir, subject_stub, redundancy, atom_stems)
    fprintf('PhiID %s (%d/%d): %s [skip existing]\n', redundancy, file_idx, n_files, file_name);
    return
end

data = load(file_path);

if isfield(data, 'time_series')
    time_series = double(data.time_series);
elseif isfield(data, 'bold_timeseries')
    time_series = double(data.bold_timeseries);
elseif isfield(data, 'source_ts')
    time_series = double(data.source_ts);
else
    error('No supported timeseries variable found in %s', file_path);
end

if size(time_series, 1) ~= 90 && size(time_series, 2) == 90
    time_series = time_series.';
end
if size(time_series, 1) ~= 90
    error('Expected 90 regions in %s, got size %dx%d', file_path, size(time_series, 1), size(time_series, 2));
end
row_std = std(time_series, 0, 2);
degenerate_rows = find(~isfinite(row_std) | row_std <= 1e-12);

n_regions = size(time_series, 1);

sts_rows = zeros(n_regions, n_regions);
rtr_rows = zeros(n_regions, n_regions);
rtx_rows = zeros(n_regions, n_regions);
rty_rows = zeros(n_regions, n_regions);
rts_rows = zeros(n_regions, n_regions);
xtr_rows = zeros(n_regions, n_regions);
xtx_rows = zeros(n_regions, n_regions);
xty_rows = zeros(n_regions, n_regions);
xts_rows = zeros(n_regions, n_regions);
ytr_rows = zeros(n_regions, n_regions);
ytx_rows = zeros(n_regions, n_regions);
yty_rows = zeros(n_regions, n_regions);
yts_rows = zeros(n_regions, n_regions);
str_rows = zeros(n_regions, n_regions);
stx_rows = zeros(n_regions, n_regions);
sty_rows = zeros(n_regions, n_regions);

fprintf('PhiID %s (%d/%d): %s\n', redundancy, file_idx, n_files, file_name);
if ~isempty(degenerate_rows)
    fprintf('  Skipping degenerate ROI rows for %s: %s\n', subject_stub, mat2str(degenerate_rows));
end

for row1 = 1:n_regions
    [sts_row, rtr_row, rtx_row, rty_row, rts_row, xtr_row, xtx_row, xty_row, ...
        xts_row, ytr_row, ytx_row, yty_row, yts_row, str_row, stx_row, sty_row] = ...
        compute_atom_row(time_series, row1, row_std, redundancy);

    sts_rows(row1, :) = sts_row;
    rtr_rows(row1, :) = rtr_row;
    rtx_rows(row1, :) = rtx_row;
    rty_rows(row1, :) = rty_row;
    rts_rows(row1, :) = rts_row;
    xtr_rows(row1, :) = xtr_row;
    xtx_rows(row1, :) = xtx_row;
    xty_rows(row1, :) = xty_row;
    xts_rows(row1, :) = xts_row;
    ytr_rows(row1, :) = ytr_row;
    ytx_rows(row1, :) = ytx_row;
    yty_rows(row1, :) = yty_row;
    yts_rows(row1, :) = yts_row;
    str_rows(row1, :) = str_row;
    stx_rows(row1, :) = stx_row;
    sty_rows(row1, :) = sty_row;
end

sts_mat = sts_rows;
rtr_mat = rtr_rows;
rtx_mat = rtx_rows;
rty_mat = rty_rows;
rts_mat = rts_rows;
xtr_mat = xtr_rows;
xtx_mat = xtx_rows;
xty_mat = xty_rows;
xts_mat = xts_rows;
ytr_mat = ytr_rows;
ytx_mat = ytx_rows;
yty_mat = yty_rows;
yts_mat = yts_rows;
str_mat = str_rows;
stx_mat = stx_rows;
sty_mat = sty_rows;

subject_meta = struct();
subject_meta.subject_stub = subject_stub;
subject_meta.redundancy = redundancy;
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

save(fullfile(output_dir, [subject_stub '__sts_mat_' redundancy '.mat']), 'sts_mat', 'subject_meta');
save(fullfile(output_dir, [subject_stub '__rtr_mat_' redundancy '.mat']), 'rtr_mat', 'subject_meta');
save(fullfile(output_dir, [subject_stub '__rtx_mat_' redundancy '.mat']), 'rtx_mat', 'subject_meta');
save(fullfile(output_dir, [subject_stub '__rty_mat_' redundancy '.mat']), 'rty_mat', 'subject_meta');
save(fullfile(output_dir, [subject_stub '__rts_mat_' redundancy '.mat']), 'rts_mat', 'subject_meta');
save(fullfile(output_dir, [subject_stub '__xtr_mat_' redundancy '.mat']), 'xtr_mat', 'subject_meta');
save(fullfile(output_dir, [subject_stub '__xtx_mat_' redundancy '.mat']), 'xtx_mat', 'subject_meta');
save(fullfile(output_dir, [subject_stub '__xty_mat_' redundancy '.mat']), 'xty_mat', 'subject_meta');
save(fullfile(output_dir, [subject_stub '__xts_mat_' redundancy '.mat']), 'xts_mat', 'subject_meta');
save(fullfile(output_dir, [subject_stub '__ytr_mat_' redundancy '.mat']), 'ytr_mat', 'subject_meta');
save(fullfile(output_dir, [subject_stub '__ytx_mat_' redundancy '.mat']), 'ytx_mat', 'subject_meta');
save(fullfile(output_dir, [subject_stub '__yty_mat_' redundancy '.mat']), 'yty_mat', 'subject_meta');
save(fullfile(output_dir, [subject_stub '__yts_mat_' redundancy '.mat']), 'yts_mat', 'subject_meta');
save(fullfile(output_dir, [subject_stub '__str_mat_' redundancy '.mat']), 'str_mat', 'subject_meta');
save(fullfile(output_dir, [subject_stub '__stx_mat_' redundancy '.mat']), 'stx_mat', 'subject_meta');
save(fullfile(output_dir, [subject_stub '__sty_mat_' redundancy '.mat']), 'sty_mat', 'subject_meta');

gradient = floor(tiedrank(mean(sts_mat))) - floor(tiedrank(mean(rtr_mat)));
save(fullfile(output_dir, [subject_stub '__sr_gradient_' redundancy '.mat']), 'gradient', 'subject_meta');

end


function [sts_row, rtr_row, rtx_row, rty_row, rts_row, xtr_row, xtx_row, xty_row, ...
    xts_row, ytr_row, ytx_row, yty_row, yts_row, str_row, stx_row, sty_row] = ...
    compute_atom_row(time_series, row1, row_std, redundancy)

n_regions = size(time_series, 1);
sts_row = zeros(1, n_regions);
rtr_row = zeros(1, n_regions);
rtx_row = zeros(1, n_regions);
rty_row = zeros(1, n_regions);
rts_row = zeros(1, n_regions);
xtr_row = zeros(1, n_regions);
xtx_row = zeros(1, n_regions);
xty_row = zeros(1, n_regions);
xts_row = zeros(1, n_regions);
ytr_row = zeros(1, n_regions);
ytx_row = zeros(1, n_regions);
yty_row = zeros(1, n_regions);
yts_row = zeros(1, n_regions);
str_row = zeros(1, n_regions);
stx_row = zeros(1, n_regions);
sty_row = zeros(1, n_regions);

for row2 = 1:n_regions
    if row1 == row2
        continue
    end

    if ~isfinite(row_std(row1)) || ~isfinite(row_std(row2)) || row_std(row1) <= 1e-12 || row_std(row2) <= 1e-12
        continue
    end

    pair_ts = [time_series(row1, :); time_series(row2, :)];
    if any(~isfinite(pair_ts), 'all')
        continue
    end

    try
        atoms = PhiIDFull(pair_ts, 1, redundancy);
    catch
        continue
    end

    if ~isstruct(atoms)
        continue
    end

    sts_row(row2) = atoms.sts;
    rtr_row(row2) = atoms.rtr;
    rtx_row(row2) = atoms.rtx;
    rty_row(row2) = atoms.rty;
    rts_row(row2) = atoms.rts;
    xtr_row(row2) = atoms.xtr;
    xtx_row(row2) = atoms.xtx;
    xty_row(row2) = atoms.xty;
    xts_row(row2) = atoms.xts;
    ytr_row(row2) = atoms.ytr;
    ytx_row(row2) = atoms.ytx;
    yty_row(row2) = atoms.yty;
    yts_row(row2) = atoms.yts;
    str_row(row2) = atoms.str;
    stx_row(row2) = atoms.stx;
    sty_row(row2) = atoms.sty;
end

end
