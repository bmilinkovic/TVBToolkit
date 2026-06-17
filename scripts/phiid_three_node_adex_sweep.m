function phiid_three_node_adex_sweep(input_dir, output_dir, measures_csv, use_parallel, n_workers)
%PHIID_THREE_NODE_ADEX_SWEEP Compute 3x3 STS/RTR matrices for 3-node AdEx sweeps.

if nargin < 4 || isempty(use_parallel)
    use_parallel = false;
end
if nargin < 5 || isempty(n_workers)
    n_workers = 0;
end

input_files = dir(fullfile(input_dir, '*.mat'));
if isempty(input_files)
    error('No input .mat files found in %s', input_dir);
end
measures = parse_measures(measures_csv);

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

fprintf('Three-node AdEx PhiID | inputs=%d | measures=%s\n', numel(input_files), strjoin(measures, ','));

if use_parallel
    parfor file_idx = 1:numel(input_files)
        process_single_file(input_files(file_idx), input_dir, output_dir, measures);
    end
else
    for file_idx = 1:numel(input_files)
        process_single_file(input_files(file_idx), input_dir, output_dir, measures);
    end
end

end


function process_single_file(file_info, input_dir, output_dir, measures)
input_path = fullfile(input_dir, file_info.name);
payload = load(input_path);
if ~isfield(payload, 'time_series')
    error('Input %s is missing time_series.', input_path);
end
time_series = double(payload.time_series);
if size(time_series, 1) ~= 3
    error('Expected 3 regions in %s, got %d.', input_path, size(time_series, 1));
end

subject_stub = extract_field_string(payload, 'stub', erase(file_info.name, '.mat'));
g_value = extract_field_scalar(payload, 'g_value', nan);
noise_value = extract_field_scalar(payload, 'noise_value', nan);
seed_value = extract_field_scalar(payload, 'seed', nan);

for measure_idx = 1:numel(measures)
    measure = lower(strtrim(measures{measure_idx}));
    output_path = fullfile(output_dir, [subject_stub '__phiid_' measure '.mat']);
    if exist(output_path, 'file') == 2
        fprintf('[skip existing] %s\n', erase(output_path, [output_dir filesep]));
        continue
    end

    sts_mat = zeros(3, 3);
    rtr_mat = zeros(3, 3);
    skipped_pairs = [];
    for i = 1:3
        for j = (i + 1):3
            pair_ts = time_series([i, j], :);
            if any(~isfinite(pair_ts(:))) || any(std(pair_ts, 0, 2) <= 0)
                skipped_pairs = [skipped_pairs; i, j]; %#ok<AGROW>
                continue
            end
            atoms = PhiIDFull(pair_ts, 1, measure);
            sts_val = extract_struct_scalar(atoms, 'sts');
            rtr_val = extract_struct_scalar(atoms, 'rtr');
            sts_mat(i, j) = sts_val;
            sts_mat(j, i) = sts_val;
            rtr_mat(i, j) = rtr_val;
            rtr_mat(j, i) = rtr_val;
        end
    end

    save(output_path, 'sts_mat', 'rtr_mat', 'subject_stub', 'measure', 'g_value', 'noise_value', 'seed_value', 'skipped_pairs', '-v7');
    fprintf('Saved %s (%s)\n', subject_stub, measure);
end
end


function measures = parse_measures(measures_csv)
if isstring(measures_csv)
    measures_csv = char(measures_csv);
end
if iscell(measures_csv)
    measures = measures_csv;
    return
end
parts = strsplit(char(measures_csv), ',');
parts = parts(~cellfun(@isempty, parts));
measures = cellfun(@strtrim, parts, 'UniformOutput', false);
end


function value = extract_field_string(payload, field_name, default_value)
if isfield(payload, field_name)
    raw = payload.(field_name);
    if iscell(raw)
        value = char(string(raw{1}));
    elseif isstring(raw)
        value = char(raw(1));
    elseif ischar(raw)
        value = raw;
    else
        value = char(string(raw(1)));
    end
else
    value = default_value;
end
end


function value = extract_field_scalar(payload, field_name, default_value)
if isfield(payload, field_name)
    raw = double(payload.(field_name));
    value = raw(1);
else
    value = default_value;
end
end


function value = extract_struct_scalar(struct_in, field_name)
if ~isstruct(struct_in) || ~isfield(struct_in, field_name)
    error('PhiID output is missing field %s.', field_name);
end
raw = double(struct_in.(field_name));
value = raw(1);
end
