import os
import pathlib

from sccnn_detection import mat_to_tf
from sccnn_detection.subpackages import NetworkOptions
from sccnn_detection import train

if os.name == 'nt':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    main_file_path = pathlib.Path(
        r'R:\tracerx\BASIS\Misc\ff\annotations-20190322\detection')
    save_path = str(pathlib.Path(r'tfrecords/ff_BASIS'))

    train_filename = 'Train-SCCNN-detection-190322'
    valid_filename = 'Valid-SCCNN-detection-190322'

    step = 10000  # Higher step is faster to process but do not increase beyond 500000. If Matlab throws can't
    # serialize error reduce the step size.

    opts = NetworkOptions.NetworkOptions(exp_dir=os.path.join(os.getcwd(), 'ExpDir-SCCNN-detection-ff-190322'),
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
                                         data_dir=save_path,
                                         train_data_filename=train_filename,
                                         valid_data_filename=valid_filename,
                                         current_epoch_num=0)

    # Do not change below

    mat_to_tf.run(main_input_path=main_file_path,
                  save_tf_path=save_path,
                  train_tf_filename=train_filename,
                  valid_tf_filename=valid_filename,
                  step=step)

    train.run(opts_in=opts)
