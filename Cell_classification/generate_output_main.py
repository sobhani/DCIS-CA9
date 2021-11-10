import sys
import os

from sccnn_classifier import save_output
from sccnn_classifier.subpackages import NetworkOptions

if os.name == 'nt':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    opts = NetworkOptions.NetworkOptions(exp_dir=r'',
                                         num_examples_per_epoch_train=1,
                                         num_examples_per_epoch_valid=1,
                                         image_height=51,
                                         image_width=51,
                                         in_feat_dim=3,
                                         in_label_dim=1,
                                         num_of_classes=5,
                                         batch_size=500,
                                         data_dir=r'',
                                         results_dir=r'',
                                         detection_results_path= r'', # detection results (CSV files including the center of each cells)
                                         current_epoch_num=0,
                                         file_name_pattern='*.svs',
                                         pre_process=True,
 					                     result_subdir='',
                                         color_code_file='Foxp3ca9.csv') #color code used in the training to define lable for each cell

    if len(sys.argv) > 1:
        opts.data_dir = sys.argv[1]

    if len(sys.argv) > 2 and opts.sub_dir_name is None:
        try:
            opts.sub_dir_name = sys.argv[2]
        except NameError:
            opts.sub_dir_name = None

    if os.path.exists(os.path.join(opts.data_dir, 'parameters-classification.txt')):
        d = {'exp_dir': opts.exp_dir,
             'results_dir': opts.results_dir,
             'file_name_pattern': opts.file_name_pattern,
             'results_subdir': opts.results_subdir,
             'in_feat_dim': opts.in_feat_dim,
             'detection_results_path': opts.detection_results_path,
             'tissue_segment_dir': opts.tissue_segment_dir,
             'num_of_classes': opts.num_of_classes,
             'color_code_file': opts.color_code_file,
             'preprocessed_dir': opts.preprocessed_dir
             }

        with open(os.path.join(opts.data_dir, 'parameters-classification.txt')) as param:
            for line in param:
                a = line.split(' ')
                d[a[0]] = a[1].strip('\n')
        print('---------------------------------------------------------------', flush=True)
        for key in d:
            print(key + ': ' + str(d[key]))
        print('---------------------------------------------------------------\n', flush=True)

        opts.exp_dir = d['exp_dir']
        opts.results_dir = d['results_dir']
        opts.file_name_pattern = d['file_name_pattern']
        opts.results_subdir = d['results_subdir']
        opts.in_feat_dim = int(d['in_feat_dim'])
        opts.detection_results_path = d['detection_results_path']
        opts.tissue_segment_dir = d['tissue_segment_dir']
        opts.num_of_classes = int(d['num_of_classes'])
        opts.color_code_file = d['color_code_file']
        opts.preprocessed_dir = d['preprocessed_dir']
        opts.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoint')
        opts.log_train_dir = os.path.join(opts.exp_dir, 'logs')

    save_output.run(opts_in=opts)
