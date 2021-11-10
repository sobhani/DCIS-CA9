clc
clear
close all;
addpath(genpath('sccnn_detection/matlab/'));
%%
results_dir = 'T:\HE_IHC_John\AEC3\Ki67\results\TTF1-180530';
data_dir = 'T:\HE_IHC_John\AEC3\Ki67\data\cws';
strength = 3;
%%
wsi_dirs=dir(fullfile(results_dir, 'network_output', 'Manualp_1_Ki67_1_HE_3*.tif'));
% wsi_dirs=wsi_dirs(~ismember({wsi_dirs.name},{'.','..', '3_TTF1.tif'}));
mapped_wsi = 'Manualp_1_HE_3_HE.tif';
mkdir(fullfile(results_dir, 'annotated_images', mapped_wsi));

for wsi_n = 1:length(wsi_dirs)
    image_path = fullfile(data_dir, mapped_wsi);
    sub_dir_name = wsi_dirs(wsi_n).name;
    files = dir(fullfile(results_dir, 'detected_points', sub_dir_name, '*.csv'));
    for i = 1:length(files)
        fprintf('%s\n', files(i).name);
        image_path_full = fullfile(image_path, [files(i).name(1:end-3), 'jpg']);                  
        detection_table = readtable(fullfile(results_dir, 'detected_points', sub_dir_name, files(i).name));
        detection = [detection_table.V2, detection_table.V3];
        
        image = imread(image_path_full);
        image = annotate_image_with_class(image, detection, [0 1 0], strength);
        imwrite(image, fullfile(results_dir, 'annotated_images', mapped_wsi, [files(i).name(1:end-3), 'png']), 'png');
    end
end