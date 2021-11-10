import tensorflow as tf
import scipy.io as sio
import os
import numpy as np
import random
import matlab.engine

from sccnn_detection.subpackages import data_utils, tools


def get_data_from_matlab(eng, size_data, size_labels, step=10000):
    assert size_data[3] == size_labels[3], "Number of data and label samples should be same."

    data = np.empty(size_data)
    labels = np.empty(size_labels)
    num_of_examples = size_data[3]
    tools.printProgressBar(0, int(num_of_examples / step) + 1, prefix='Copying Progress:', suffix='Complete', length=50)

    # set dtype for data and labels
    eng.eval("images.data = (images.data)./255;", nargout=0)
    eng.eval("data_temp = images.data(:,:,:,1);", nargout=0)
    data_temp = eng.workspace['data_temp']
    data_temp = np.array(data_temp._data).reshape(data_temp.size, order='F')

    eng.eval("labels_temp = images.labels(:,:,:,1);", nargout=0)
    labels_temp = eng.workspace['labels_temp']
    labels_temp = np.array(labels_temp._data).reshape(labels_temp.size, order='F')

    eng.eval("set = images.set;", nargout=0)
    images_set = eng.workspace['set']
    images_set = np.array(images_set._data).reshape(images_set.size, order='F')
    images_set = np.squeeze(images_set).astype('bool')

    if data.dtype != data_temp.dtype:
        data = data.astype(data_temp.dtype)

    if labels.dtype != labels_temp.dtype:
        labels = labels.astype(labels_temp.dtype)

    for i in range(0, int(num_of_examples / step) + 1):
        start = i * step + 1
        end = (i + 1) * step
        if end > num_of_examples:
            end = num_of_examples
        tools.printProgressBar(i + 1, int(num_of_examples / step) + 1,
                               prefix='Copying Progress:', suffix='Complete', length=50)
        eng.eval("data_temp = images.data(:,:,:," + str(start) + ":" + str(end) + ");", nargout=0)
        data_temp = eng.workspace['data_temp']
        data_temp = np.array(data_temp._data).reshape(data_temp.size, order='F')
        if data_temp.ndim < 3:
            data_temp = np.expand_dims(data_temp, axis=2)
        data[:, :, :, start - 1:end] = data_temp

        eng.eval("labels_temp = images.labels(:,:,:," + str(start) + ":" + str(end) + ");", nargout=0)
        labels_temp = eng.workspace['labels_temp']
        labels_temp = np.array(labels_temp._data).reshape(labels_temp.size, order='F')
        if labels_temp.ndim < 3:
            labels_temp = np.expand_dims(labels_temp, axis=2)
        labels[:, :, :, start - 1:end] = labels_temp

    eng.clear("all", nargout=0)
    return data, labels, images_set


def get_imdb_size(eng, imdb_path):
    eng.load(imdb_path, nargout=0)

    size_data = eng.eval("size(images.data)")
    size_labels = eng.eval("size(images.labels)")

    size_data = np.array(size_data[0], int)
    size_labels = np.array(size_labels[0], int)

    return size_data, size_labels


def load_full_data(eng, main_file_path, step=10000):
    imdb_data_warwick_format = str(main_file_path.joinpath('imdb.mat'))
    print('Reading: ' + imdb_data_warwick_format)

    size_data, size_labels = get_imdb_size(eng, imdb_path=imdb_data_warwick_format)
    full_data, full_labels, images_set = get_data_from_matlab(eng, size_data, size_labels, step=step)

    return full_data, full_labels, images_set


def run(main_input_path, save_tf_path, train_tf_filename, valid_tf_filename, step=10000):

    if not os.path.exists(save_tf_path):
        os.makedirs(save_tf_path, exist_ok=True)

    eng = matlab.engine.start_matlab()
    full_data, full_labels, images_set = load_full_data(eng, main_file_path=main_input_path, step=step)

    # Training Set
    train_full_data = full_data[:, :, :, images_set == True]
    train_full_labels = full_labels[:, :, :, images_set == True]

    files = list(range(0, train_full_data.shape[3]))
    random.shuffle(files)
    print('Writing: ', os.path.join(save_tf_path, train_tf_filename + '.tfrecords'))
    train_writer = tf.python_io.TFRecordWriter(os.path.join(save_tf_path, train_tf_filename + '.tfrecords'))
    num_examples = 0
    tools.printProgressBar(0, len(files), prefix='Progress:', suffix='Complete', length=50)
    for file_n in range(len(files)):
        tools.printProgressBar(file_n + 1, len(files), prefix='Progress:', suffix='Complete', length=50)
        data = train_full_data[:, :, :, files[file_n]]
        labels = train_full_labels[:, :, :, files[file_n]]
        tf_serialized_example = data_utils.encode(in_feat=data, labels=labels)
        train_writer.write(tf_serialized_example)
        num_examples += 1

    out_dict = {'num_examples': num_examples}
    sio.savemat(os.path.join(save_tf_path, train_tf_filename + '.mat'), out_dict)
    train_writer.close()

    del train_full_data, train_full_labels

    # Validation Set
    valid_full_data = full_data[:, :, :, images_set == False]
    valid_full_labels = full_labels[:, :, :, images_set == False]

    files = list(range(0, valid_full_data.shape[3]))
    random.shuffle(files)
    print('Writing: ', os.path.join(save_tf_path, valid_tf_filename + '.tfrecords'))
    valid_writer = tf.python_io.TFRecordWriter(os.path.join(save_tf_path, valid_tf_filename + '.tfrecords'))
    num_examples = 0
    tools.printProgressBar(0, len(files), prefix='Progress:', suffix='Complete', length=50)
    for file_n in range(len(files)):
        tools.printProgressBar(file_n + 1, len(files), prefix='Progress:', suffix='Complete', length=50)
        data = valid_full_data[:, :, :, files[file_n]]
        labels = valid_full_labels[:, :, :, files[file_n]]
        tf_serialized_example = data_utils.encode(in_feat=data, labels=labels)
        valid_writer.write(tf_serialized_example)
        num_examples += 1

    out_dict = {'num_examples': num_examples}
    sio.savemat(os.path.join(save_tf_path, valid_tf_filename + '.mat'), out_dict)
    valid_writer.close()
