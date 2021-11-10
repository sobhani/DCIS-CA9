function save_detection_output(results_dir, sub_dir_name, mat_file_name, image_path_full, maxclique)
    if ~exist(...
            fullfile(results_dir, 'annotated_images', sub_dir_name, ...
            [mat_file_name(1:end-3), 'png']), 'file')
        strength = 3;
   
        output = load(fullfile(results_dir, 'network_output', sub_dir_name, mat_file_name));
        output = output.output;
        image = imread(image_path_full);
        grayImage1 = min(im2uint8(image), [], 3);
        grayImage2 = max(im2uint8(image), [], 3);
        thgray = (grayImage1<200) & (grayImage2>0);
        output = output.*thgray;
        
        detection = FindLocalMaximaMaxClique(output,maxclique.distance,maxclique.threshold);
        if ~isempty(detection)
            linearInd = sub2ind(size(thgray), detection(:,2), detection(:,1));
            detection(~thgray(linearInd),:) = [];
        end
        V = cell(size(detection,1),3);
        detection_table = cell2table(V);
        if ~isempty(detection)
            detection_table.V1 = repmat({'None'},[size(detection,1),1]);
            detection_table.V2 = detection(:,1);
            detection_table.V3 = detection(:,2);
            writetable(detection_table, fullfile(results_dir, 'detected_points', sub_dir_name, [mat_file_name(1:end-3), 'csv']));
            image = annotate_image_with_class(image, detection, [0 1 0], strength);
        else
            fileID = fopen(fullfile(results_dir, 'detected_points', sub_dir_name, [mat_file_name(1:end-3), 'csv']), 'w');
            fprintf(fileID, 'V1,V2,V3');
        end
        
        imwrite(image, fullfile(results_dir, 'annotated_images', sub_dir_name, [mat_file_name(1:end-3), 'png']), 'png');
        save(fullfile(results_dir, 'network_output', sub_dir_name, mat_file_name),...
            'output', 'detection', '-v7.3');
        close all;
    else
        fprintf('Already Processed %s\n', ...
            fullfile(results_dir, 'annotated_images', sub_dir_name, ...
            [mat_file_name(1:end-3), 'png']))
    end
end