import pathlib
import os
import time
from datetime import datetime
import tensorflow as tf
from sccnn_classifier import mat_to_tf, sccnn_classifier
from sccnn_classifier.subpackages import NetworkOptions, data_utils
import scipy.io as sio
import pandas as pd


if os.name=='nt':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run(test_name, main_file_path,
        checkpoint_path='checkpoint',
        tfrecord_savepath='tfrecords',
        save_path='evaluation_results',
        tfrecord_name=None):

    if tfrecord_name is None:
        tfrecord_name = test_name

    if not os.path.exists(tfrecord_savepath):
        os.makedirs(tfrecord_savepath)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    test_files = list(main_file_path.joinpath('test').glob('*.mat'))
    print('-------------------------------------------------------------', flush=True)
    print('Saving ' + tfrecord_savepath + '/' + test_name, flush=True)
    print('-------------------------------------------------------------', flush=True)
    mat_to_tf.write_to_tf(files=test_files, save_path=tfrecord_savepath, save_filename=test_name)

    opts = NetworkOptions.NetworkOptions(exp_dir=checkpoint_path,
                                         num_examples_per_epoch_train=1,
                                         num_examples_per_epoch_valid=1,
                                         image_height=51,
                                         image_width=51,
                                         in_feat_dim=3,
                                         in_label_dim=1,
                                         num_of_classes=5,
                                         batch_size=1000,
                                         num_of_epoch=500,
                                         data_dir=tfrecord_savepath,
                                         train_data_filename=tfrecord_name,
                                         current_epoch_num=0)

    network = sccnn_classifier.SccnnClassifier(batch_size=opts.batch_size,
                                               image_height=opts.image_height,
                                               image_width=opts.image_width,
                                               in_feat_dim=opts.in_feat_dim,
                                               in_label_dim=opts.in_label_dim,
                                               num_of_classes=opts.num_of_classes,
                                               tf_device=opts.tf_device)

    param = sio.loadmat(os.path.join(opts.data_dir, opts.train_data_filename + '.mat'))
    test_num_examples = param['num_examples'][0][0]

    test_dataset = data_utils.get_data_set(
        filename=os.path.join(opts.data_dir, opts.train_data_filename + '.tfrecords'),
        num_epochs=opts.num_of_epoch,
        shuffle_size=opts.batch_size * 10,
        batch_size=opts.batch_size,
        prefetch_buffer=opts.batch_size * 2)

    network.run_checks(opts=opts)

    iterator = tf.data.Iterator.from_structure(test_dataset.output_types,
                                               test_dataset.output_shapes)
    images, labels = iterator.get_next()
    test_init_op = iterator.make_initializer(test_dataset)

    logits, _ = network.inference(images=images, is_training=False)
    labels = tf.one_hot(tf.squeeze(labels, axis=1), opts.num_of_classes)

    loss = network.loss_function(logits=logits, labels=labels, weighted_loss_per_class=opts.weighted_loss_per_class)
    _, accuracy = tf.metrics.accuracy(tf.argmax(labels, 1), tf.argmax(logits, 1))

    test_count = int((test_num_examples / opts.batch_size) + 1)

    config = tf.ConfigProto()
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=opts.num_of_epoch)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(opts.checkpoint_dir)
        assert ckpt, "No Checkpoint file found"
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Checkpoint file found at ' + ckpt.model_checkpoint_path, flush=True)

        sess.run(test_init_op)

        try:
            avg_loss = 0.0
            avg_accuracy = 0.0
            for step in range(test_count):
                start_time_step = time.time()
                loss_value, accuracy_value = \
                    sess.run([loss, accuracy])
                duration = time.time() - start_time_step
                format_str = (
                    '%s: step %d/ %d, Training Loss = %.2f, Training Accuracy = %.2f, (%.2f sec/step)')
                print(
                    format_str % (
                        datetime.now(), step + 1, test_count,
                        loss_value, accuracy_value, float(duration)), flush=True)

                avg_loss = avg_loss + loss_value
                avg_accuracy = avg_accuracy + accuracy_value

        except tf.errors.OutOfRangeError:
            print('Done', flush=True)

        results = {'Loss': avg_loss/(step + 1),
                   'Accuracy': avg_accuracy/(step + 1)}

        print(results, flush=True)

        df = pd.DataFrame(results, index=[0])
        df.to_csv(save_path + '/' + test_name + '.csv', index=False)


if __name__ == '__main__':
    test_name = 'SCCNN_51x51_5-fold_cv1'
    main_file_path = pathlib.Path(r'/mnt/scratch/users/fsobhani/New_Classifier/20190522_SCCNNClassifier/mat/cv1')
    checkpoint_path = str(pathlib.Path(r'SCCNN_51x51_5-fold_cv1'))

    run(test_name, main_file_path, checkpoint_path=checkpoint_path)
