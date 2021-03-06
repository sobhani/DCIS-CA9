import os
import pathlib


from sccnn_classifier.subpackages import NetworkOptions
from sccnn_classifier import train, mat_to_tf


if os.name=='nt':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    save_tf_path = pathlib.Path(r'.\tfrecords')
    main_input_path = pathlib.Path(r'') # all input data mat files 
    train_tf_filename = 'TrainData'
    valid_tf_filename = 'validData'



    opts = NetworkOptions.NetworkOptions(exp_dir=os.path.normpath(os.path.join(os.getcwd(), r'EXPDir')),
                                         num_examples_per_epoch_train=1,
                                         num_examples_per_epoch_valid=1,
                                         image_height=51,
                                         image_width=51,
                                         in_feat_dim=3,
                                         in_label_dim=1,
                                         num_of_classes=5,
                                         batch_size=1000,
                                         num_of_epoch=500,
                                         data_dir=save_tf_path,
                                         train_data_filename=train_tf_filename,
                                         valid_data_filename=valid_tf_filename,
                                         current_epoch_num=0)

    # Do not change anything below
#
    mat_to_tf.run(main_input_path=main_input_path,
                  save_tf_path=save_tf_path,
                  train_tf_filename=train_tf_filename,
                  valid_tf_filename=valid_tf_filename)

    train.run(opts_in=opts)
