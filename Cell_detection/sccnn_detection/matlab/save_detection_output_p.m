function save_detection_output_p(results_dir, sub_dir_name, image_path, maxclique)
    files = dir(fullfile(results_dir, 'network_output', sub_dir_name, '*.mat'));
    parfor i = 1:length(files)
        fprintf('%s\n', files(i).name);
        mat_file_name = files(i).name;
        image_path_full = fullfile(image_path, [files(i).name(1:end-3), 'jpg']);
        save_detection_output(results_dir, sub_dir_name, mat_file_name, image_path_full, maxclique);
    end
end