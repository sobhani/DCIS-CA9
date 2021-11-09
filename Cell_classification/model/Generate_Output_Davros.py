import os
import step4_cell_class.sccnn_classifier as sccnn_classifier
from step4_cell_class.subpackages import NetworkOptions
import tensorflow as tf
tf.reset_default_graph()

# data_dir = sys.argv[1]
# sub_dir_name = sys.argv[2]
#
# d = {'tissue_segment_dir': '',
#      'preprocessed_dir': None,
#      'exp_dir': 'ExpDir',
#      'num_of_classes': 4}
# with open(os.path.join(data_dir, 'parameters-classification.txt')) as param:
#     for line in param:
#         a = line.split(' ')
#         d[a[0]] = a[1].strip('\n')
#
# print('results_dir: ' + d['results_dir'], flush=True)
# print('tissue_segment_dir: ' + d['tissue_segment_dir'], flush=True)
# print('detection_results_path:' + d['detection_results_path'], flush=True)
# print('file_name_pattern: ' + d['file_name_pattern'], flush=True)
# print('date: ' + d['date'], flush=True)
# print('exp_dir: ' + d['exp_dir'], flush=True)
# print('color_code_file:' + d['color_code_file'], flush=True)

def run_cell_classification(data_dir,
                            sub_dir_name,
                            in_feat_dim,
                            num_of_classes,
                            results_dir,
                            detection_results_path,
                            tissue_segment_dir,
                            preprocessed_dir,
                            file_name_pattern,
                            color_code_file='step4_cell_seg/colorcodes/HE_Fib_Lym_Tum_Others.csv',
                            exp_dir='step4_cell_seg/ExpDir-HE-4class',
                            minimal_output=False):
    
    opts = NetworkOptions.NetworkOptions(exp_dir=exp_dir,
                                         num_examples_per_epoch_train=1,
                                         num_examples_per_epoch_valid=1,
                                         image_height=51,
                                         image_width=51,
                                         in_feat_dim=int(in_feat_dim),
                                         in_label_dim=1,
                                         num_of_classes=int(num_of_classes),
                                         batch_size=500,
                                         data_dir=data_dir,
                                         results_dir=results_dir,
                                         detection_results_path=detection_results_path,
                                         tissue_segment_dir=tissue_segment_dir,
                                         preprocessed_dir=preprocessed_dir,
                                         current_epoch_num=0,
                                         file_name_pattern=file_name_pattern,
                                         pre_process=True,
                                         color_code_file=color_code_file,
                                         minimal_output=minimal_output)
    
    # I uncommented this because we agreed we don't want date
    # opts.results_dir = os.path.join(opts.results_dir, d['date'])
    # opts.preprocessed_dir = os.path.join(opts.preprocessed_dir, d['date'])
    
    if not os.path.isdir(opts.results_dir):
        os.makedirs(opts.results_dir)
    if not os.path.isdir(os.path.join(opts.results_dir, 'mat')):
        os.makedirs(os.path.join(opts.results_dir, 'mat'))
    if (not opts.minimal_output) and (not os.path.isdir(os.path.join(opts.results_dir, 'annotated_images'))):
        os.makedirs(os.path.join(opts.results_dir, 'annotated_images'))
    if not os.path.isdir(os.path.join(opts.results_dir, 'csv')):
        os.makedirs(os.path.join(opts.results_dir, 'csv'))
    if not os.path.isdir(os.path.join(opts.preprocessed_dir, 'pre_processed')):
        os.makedirs(os.path.join(opts.preprocessed_dir, 'pre_processed'))
    
    Network = sccnn_classifier.SccnnClassifier(batch_size=opts.batch_size,
                                               image_height=opts.image_height,
                                               image_width=opts.image_width,
                                               in_feat_dim=opts.in_feat_dim,
                                               in_label_dim=opts.in_label_dim,
                                               num_of_classes=opts.num_of_classes)
    
    # print('\n\n\n', flush=True)
    # print('opts.data_dir:' + os.path.join(opts.data_dir, sub_dir_name), flush=True)
    # print('opts.results_dir:' + os.path.join(opts.results_dir, sub_dir_name), flush=True)
    # print('opts.detection_results_path:' + os.path.join(opts.detection_results_path, sub_dir_name), flush=True)
    # print('opts.preprocessed_dir:' + os.path.join(opts.preprocessed_dir, 'pre_processed', sub_dir_name), flush=True)
    # print('opts.tissue_segmentation:' + os.path.join(opts.tissue_segment_dir, sub_dir_name), flush=True)
    # print('opts.file_name_pattern:' + opts.file_name_pattern, flush=True)
    # print('opts.pre_process:' + str(opts.pre_process), flush=True)
    # print('opts.exp_dir:' + opts.exp_dir, flush=True)
    # print('color_code_file:' + color_code_file, flush=True)
    
    Network.generate_output_sub_dir(opts=opts, sub_dir_name=sub_dir_name, network_output=True)
