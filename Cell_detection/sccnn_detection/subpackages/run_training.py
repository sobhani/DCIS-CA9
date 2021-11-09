import tensorflow as tf
import numpy as np
import os
import scipy.io as sio
import time
from datetime import datetime
from sccnn_detection.subpackages import data_utils


def run_training(network, opts):
    param = sio.loadmat(os.path.join(opts.data_dir, opts.train_data_filename + '.mat'))
    train_num_examples = param['num_examples'][0][0]

    param = sio.loadmat(os.path.join(opts.data_dir, opts.valid_data_filename + '.mat'))
    valid_num_examples = param['num_examples'][0][0]

    training_dataset = data_utils.get_data_set(
        filename=os.path.join(opts.data_dir, opts.train_data_filename + '.tfrecords'),
        num_epochs=opts.num_of_epoch,
        shuffle_size=opts.batch_size*5,
        batch_size=opts.batch_size,
        prefetch_buffer=opts.batch_size * 5)

    validation_dataset = data_utils.get_data_set(
        filename=os.path.join(opts.data_dir, opts.valid_data_filename + '.tfrecords'),
        num_epochs=opts.num_of_epoch,
        shuffle_size=opts.batch_size*5,
        batch_size=opts.batch_size,
        prefetch_buffer=opts.batch_size * 5)

    network.run_checks(opts=opts)
    iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                               training_dataset.output_shapes)
    data, labels = iterator.get_next()
    training_init_op = iterator.make_initializer(training_dataset)
    validation_init_op = iterator.make_initializer(validation_dataset)

    pad_val_h = int((opts.image_height - opts.label_height)/2)
    pad_val_w = int((opts.image_width - opts.label_width) / 2)
    paddings = tf.constant([[0, 0], [pad_val_h, pad_val_h], [pad_val_w, pad_val_w], [0, 0]])
    labels = tf.pad(labels, paddings, "CONSTANT")

    data, labels = data_utils.augment(data_in=data, labels_in=labels, in_feat_dim=opts.in_feat_dim)
    data = data_utils.random_variations(data_in=data, in_feat_dim=opts.in_feat_dim, num_feat_aug=1)
    labels = tf.image.crop_to_bounding_box(labels, pad_val_h, pad_val_w, opts.label_height, opts.label_width)

    global_step = tf.Variable(1.0, trainable=False)
    network.LearningRate = tf.placeholder(tf.float32)
    logits = network.inference(images=data)
    loss = network.loss_function(logits=logits, labels=labels)
    train_op = network.train(loss=loss)

    imr0 = logits[0:1, :, :, 0:1]
    imr1 = data[0:1, :, :, 1:4]
    imr2 = labels[0:1, :, :, 0:1]
    _ = tf.summary.image('Output_1', imr0)
    _ = tf.summary.image('Input_1', imr1)
    _ = tf.summary.image('label', imr2)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    avg_training_loss = 0.0
    avg_validation_loss = 0.0
    training_loss = 50000.0
    validation_loss = 50000.0

    train_count = int((train_num_examples / opts.batch_size) + 1)
    valid_count = int((valid_num_examples / opts.batch_size) + 1)

    with tf.Session() as sess:
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(opts.log_train_dir, 'train'), sess.graph)
        valid_writer = tf.summary.FileWriter(os.path.join(opts.log_train_dir, 'valid'), sess.graph)
        init = tf.global_variables_initializer()
        ckpt = tf.train.get_checkpoint_state(opts.checkpoint_dir)
        curr_epoch = 1
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]) + 1
            curr_epoch = int(global_step)
            print('Checkpoint file found at ' + ckpt.model_checkpoint_path, flush=True)
            workspace = sio.loadmat(os.path.join(opts.exp_dir, 'avg_training_loss.mat'))
            avg_training_loss = np.array(workspace['avg_training_loss'])
            avg_validation_loss = np.array(workspace['avg_validation_loss'])
            training_loss = avg_training_loss[0, -1]
            validation_loss = avg_validation_loss[0, -1]
        else:
            sess.run(init)
            print('No checkpoint file found', flush=True)

        for epoch in range(curr_epoch, opts.num_of_epoch+1):
            lr = 0.001
            opts.current_epoch_num = global_step
            start_time_epoch = time.time()
            sess.run(training_init_op)
            step = 0
            try:
                avg_loss = 0.0
                for step in range(train_count):
                    start_time_step = time.time()
                    if (step % int(10) == 0) or (step == 0) or (step == train_count-1):
                        summary_str, _, loss_value, logits_out = \
                            sess.run([summary_op, train_op, loss, logits],
                                     feed_dict={network.LearningRate: lr})
                        train_writer.add_summary(summary_str, step + epoch * train_count)
                        duration = time.time() - start_time_step
                        format_str = (
                            '%s: epoch %d, step %d/ %d, Training Loss = %.2f, (%.2f sec/step)')
                        print(
                            format_str % (
                                datetime.now(), epoch, step + 1, train_count, loss_value, float(duration)),
                            flush=True)
                    else:
                        _, loss_value = sess.run([train_op, loss],
                                                 feed_dict={network.LearningRate: lr})

                    avg_loss += loss_value

                training_loss = avg_loss / train_count

            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' % (epoch, step), flush=True)

            sess.run(validation_init_op)
            step = 0
            try:
                avg_loss = 0.0
                for step in range(valid_count):
                    start_time_step = time.time()
                    if (step % int(5) == 0) or (step == 0) or (step == valid_count-1):
                        summary_str, loss_value, logits_out = \
                            sess.run([summary_op, loss, logits])
                        valid_writer.add_summary(summary_str, (step * train_count/valid_count) + (epoch * train_count))
                        duration = time.time() - start_time_step
                        format_str = (
                            '%s: epoch %d, step %d/ %d, Validation Loss = %.2f, (%.2f sec/step)')
                        print(
                            format_str % (
                                datetime.now(), epoch, step + 1, valid_count, loss_value, float(duration)),
                            flush=True)
                    else:
                        loss_value, logits_out = sess.run([loss, logits])

                    avg_loss += loss_value

                validation_loss = avg_loss / valid_count

            except tf.errors.OutOfRangeError:
                print('Done validation for %d epochs, %d steps.' % (epoch, step), flush=True)

            # Save the model after each epoch.
            checkpoint_path = os.path.join(opts.checkpoint_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=global_step)
            global_step = global_step + 1
            # Average loss on training and validation data sets.
            duration = time.time() - start_time_epoch
            format_str = (
                '%s: epoch %d, Training Loss = %.2f, Validation Loss = %.2f, (%.2f sec/epoch)')
            print(format_str % (
                datetime.now(), epoch, training_loss, validation_loss, float(duration)), flush=True)
            if epoch == 1:
                avg_training_loss = [int(training_loss)]
                avg_validation_loss = [int(validation_loss)]
            else:
                avg_training_loss = np.append(avg_training_loss, [int(training_loss)])
                avg_validation_loss = np.append(avg_validation_loss, [int(validation_loss)])
            avg_training_loss_dict = {'avg_training_loss': avg_training_loss,
                                      'avg_validation_loss': avg_validation_loss}
            sio.savemat(file_name=os.path.join(opts.exp_dir, 'avg_training_loss.mat'), mdict=avg_training_loss_dict)
            print(avg_training_loss, flush=True)
            print(avg_validation_loss, flush=True)

    return avg_training_loss
