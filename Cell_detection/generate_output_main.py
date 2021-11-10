import sys
import os

from sccnn_detection.subpackages import NetworkOptions
from sccnn_detection import save_output

if os.name == 'nt':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':

    opts = NetworkOptions.NetworkOptions(exp_dir=r'./ExpDir-SCCNN-detection-foxp3-13-15102019',
                                         num_examples_per_epoch_train=1,
                                         num_examples_per_epoch_valid=1,
                                         image_height=31,
                                         image_width=31,
                                         in_feat_dim=4,
                                         label_height=13,
                                         label_width=13,
                                         in_label_dim=1,
                                         batch_size=88,
                                         data_dir=r'Z:\DCIS_Duke_Faranak\HDD\Foxp3ca9-Final\data\cws\tmp',
                                         results_dir=r'Z:\DCIS_Duke_Faranak\HDD\Foxp3ca9-Final\data\10112021-testpipline',
                                         tissue_segment_dir='',
                                         file_name_pattern='*.svs',
                                         maxclique_distance=8,
                                         maxclique_threshold=0.15,
                                         feat_set=['h', 'rgb'],
                                         pre_process=True,
                                         results_subdir='10122019',
                                         )

    if len(sys.argv) > 1:
        opts.data_dir = sys.argv[1]

    if len(sys.argv) > 2 and opts.sub_dir_name is None:
        try:
            opts.sub_dir_name = sys.argv[2]
        except NameError:
            opts.sub_dir_name = None

    if os.path.exists(os.path.join(opts.data_dir, 'parameters-detection22.txt')):
        d = {'exp_dir': opts.exp_dir,
             'results_dir': opts.results_dir,
             'results_subdir': opts.results_subdir,
             'file_name_pattern': opts.file_name_pattern,
             'in_feat_dim': opts.in_feat_dim,
             'tissue_segment_dir': opts.tissue_segment_dir,
             'preprocessed_dir': opts.preprocessed_dir
             }

        with open(os.path.join(opts.data_dir, 'parameters-detection.txt')) as param:
            for line in param:
                a = line.split(' ')
                d[a[0]] = a[1].strip('\n')
        print('---------------------------------------------------------------', flush=True)
        for key in d:
            print(key + ': ' + str(d[key]))
        print('---------------------------------------------------------------\n', flush=True)

        opts.exp_dir = d['exp_dir']
        opts.results_dir = d['results_dir']
        opts.results_subdir = d['results_subdir']
        opts.file_name_pattern = d['file_name_pattern']
        opts.in_feat_dim = int(d['in_feat_dim'])
        opts.tissue_segment_dir = d['tissue_segment_dir']
        opts.preprocessed_dir = d['preprocessed_dir']

    save_output.run(opts_in=opts)
