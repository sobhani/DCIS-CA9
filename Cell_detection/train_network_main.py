import os


from sccnn_detection.subpackages import NetworkOptions
from sccnn_detection import train

if os.name == 'nt':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    opts = NetworkOptions.NetworkOptions(exp_dir=os.path.join(os.getcwd(), r'.\ExpDir_CA9FOXp3'),
                                         num_examples_per_epoch_train=1,
                                         num_examples_per_epoch_valid=1,
                                         image_height=31,
                                         image_width=31,
                                         in_feat_dim=4,
                                         label_height=13,
                                         label_width=13,
                                         in_label_dim=1,
                                         batch_size=1000,
                                         num_of_epoch=500,
                                         data_dir=r'tfrecords/foxp2ca9-13',
                                         train_data_filename='fTrain-SCCNN-detection-24102019',
                                         valid_data_filename='fValid-SCCNN-detection-24102019',
                                         current_epoch_num=0)
    train.run(opts_in=opts)
