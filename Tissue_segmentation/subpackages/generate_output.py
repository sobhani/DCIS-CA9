import tensorflow as tf
import glob
import os
import numpy as np
import scipy.io as sio
import time
import cv2
from datetime import datetime
import pandas as pd
import math

from subpackages import Patches


def make_sub_dirs(opts, sub_dir_name):
    if not os.path.isdir(os.path.join(opts.results_dir, 'mat', sub_dir_name)):
        os.makedirs(os.path.join(opts.results_dir, 'mat', sub_dir_name))

    if not os.path.isdir(os.path.join(opts.results_dir, 'pre_processed', sub_dir_name)):
        os.makedirs(os.path.join(opts.results_dir, 'pre_processed', sub_dir_name))

    if (not opts.minimal_output) and (not os.path.isdir(os.path.join(opts.results_dir, 'csv'))):
        os.makedirs(os.path.join(opts.results_dir, 'csv'))

    if (not opts.minimal_output) and (not os.path.isdir(os.path.join(opts.results_dir, 'annotated_images'))):
        os.makedirs(os.path.join(opts.results_dir, 'annotated_images'))

    if not os.path.isdir(os.path.join(opts.results_dir, 'mask')):
        os.makedirs(os.path.join(opts.results_dir, 'mask'))


def pre_process_images(opts, sub_dir_name):
    make_sub_dirs(opts, sub_dir_name)
    
    if opts.pre_process:
        files = sorted(glob.glob(os.path.join(opts.data_dir, sub_dir_name, 'Ss1.jpg')))
        
        # print('Pre-Processing %s\n' % sub_dir_name)
            
        for i in range(len(files)):
            if not os.path.isfile(os.path.join(opts.results_dir, 'pre_processed', sub_dir_name, os.path.basename(files[i][:-3]) + 'mat')):
                #target_image = np.float32(cv2.cvtColor(cv2.imread('TargetImage.png'), cv2.COLOR_BGR2RGB))/255.0
                target_image = np.float32(cv2.cvtColor(cv2.imread(os.path.join(os.path.dirname(__file__), '..', 'TargetImage.png')), cv2.COLOR_BGR2RGB))/255.0
                image = np.float32(cv2.cvtColor(cv2.imread(os.path.join(opts.data_dir, sub_dir_name, os.path.basename(files[i]))), cv2.COLOR_BGR2RGB))/255.0
                feat = 255.0*norm_reinhard(image, target_image)
                feat[feat < 0.0] = 0.0
                feat[feat > 255.0] = 255.0
                feat = np.round(feat)
                sio.savemat(os.path.join(opts.results_dir, 'pre_processed', sub_dir_name, os.path.basename(files[i][:-3]) + 'mat'), {'matlab_output': {'feat': feat}})
                
def norm_reinhard(source_image, target_image):
    source_lab = cv2.cvtColor(source_image, cv2.COLOR_RGB2Lab)

    ms = np.mean(source_lab, axis=(0, 1))
    stds = np.std(source_lab, axis=(0, 1))

    target_lab = cv2.cvtColor(target_image, cv2.COLOR_RGB2Lab)
    
    mt = np.mean(target_lab, axis=(0, 1))
    stdt = np.std(target_lab, axis=(0, 1))

    norm_lab = np.copy(source_lab)

    norm_lab[:,:,0] = ((norm_lab[:,:,0]-ms[0])*(stdt[0]/stds[0]))+mt[0]
    norm_lab[:,:,1] = ((norm_lab[:,:,1]-ms[1])*(stdt[1]/stds[1]))+mt[1]
    norm_lab[:,:,2] = ((norm_lab[:,:,2]-ms[2])*(stdt[2]/stds[2]))+mt[2]

    norm_image = cv2.cvtColor(norm_lab, cv2.COLOR_Lab2RGB)
    
    return norm_image

def generate_network_output(opts, sub_dir_name, network, sess,
                            logits):

    make_sub_dirs(opts, sub_dir_name)
    image_path = os.path.join(opts.data_dir, sub_dir_name)
    files = sorted(glob.glob(os.path.join(opts.data_dir, sub_dir_name, 'Ss1.jpg')))
    for i in range(len(files)):
        # print(files[i])
        image_path_full = os.path.join(opts.data_dir, sub_dir_name, files[i])
        if opts.pre_process:
            workspace = sio.loadmat(os.path.join(opts.results_dir, 'pre_processed', sub_dir_name,
                                                 os.path.basename(files[i][:-3]) + 'mat'))
            matlab_output = workspace['matlab_output']
            feat = np.array(matlab_output['feat'][0][0])
        else:
            feat = image_path_full

        patch_obj = Patches.Patches(
            img_patch_h=opts.image_height, img_patch_w=opts.image_width,
            stride_h=opts.stride_h, stride_w=opts.stride_w,
            label_patch_h=opts.label_height, label_patch_w=opts.label_width)

        image_patches = patch_obj.extract_patches(feat)
        image_patches = image_patches.astype(np.float32, copy=False)/255.0
            
        opts.num_examples_per_epoch_for_train, opts.image_height, opts.image_width, opts.in_feat_dim = \
            image_patches.shape
        label_patches = np.zeros([opts.num_examples_per_epoch_for_train, opts.label_height,
                                  opts.label_width, opts.num_of_classes], dtype=np.float32)

        data_train = np.zeros((opts.batch_size, opts.image_height, opts.image_width, opts.in_feat_dim), dtype=np.float32)

        start_time = time.time()
        
        for start in range(0, opts.num_examples_per_epoch_for_train, opts.batch_size):
            end = min(start + opts.batch_size, opts.num_examples_per_epoch_for_train)
                
            data_train[:(end-start), :, :, :] = image_patches[start:end, :, :, :]
                
            logits_out = sess.run(
                logits,
                feed_dict={network.images: data_train,
                           })
            label_patches[start:end, :, :, :] = logits_out[:(end-start), :, :, :]

        output = patch_obj.merge_patches(label_patches)
        mat = {'output': output}
        mat_file_name = os.path.basename(files[i][:-3]) + 'mat'
        sio.savemat(os.path.join(opts.results_dir, 'mat', sub_dir_name, mat_file_name), mat)

        duration = time.time() - start_time
        format_str = (
            '%s: file %d/ %d, (%.2f sec/file)')
        # print(format_str % (datetime.now(), i + 1, len(files), float(duration)))

        save_detection_output_p(opts, sub_dir_name, image_path)

        workspace = sio.loadmat(os.path.join(opts.results_dir, 'mat', sub_dir_name,
                                             os.path.basename(files[i][:-3]) + 'mat'))
        mat = workspace['mat']
        bin_label = mat['BinLabel'][0][0]
        bin_label = bin_label.astype('bool')
        slide_h = bin_label.shape[0]
        slide_w = bin_label.shape[1]
        cws_h = 125
        cws_w = 125
        iter_tot = 0
        cws_file = []
        has_tissue = []
        for h in range(int(math.ceil((slide_h - cws_h) / cws_h + 1))):
            for w in range(int(math.ceil((slide_w - cws_w) / cws_w + 1))):
                start_h = h * cws_h
                end_h = (h * cws_h) + cws_h
                start_w = w * cws_w
                end_w = (w * cws_w) + cws_w
                if end_h > slide_h:
                    end_h = slide_h

                if end_w > slide_w:
                    end_w = slide_w

                cws_file.append('Da' + str(iter_tot))
                curr_bin_label = bin_label[start_h:end_h, start_w:end_w]
                has_tissue.append(curr_bin_label.any())
                if curr_bin_label.any():
                    mat = {'bin_label': curr_bin_label}
                    sio.savemat(os.path.join(opts.results_dir, 'mat', sub_dir_name,
                                             cws_file[iter_tot] + '.mat'), mat)

                iter_tot = iter_tot + 1

        if not opts.minimal_output:
            data_dict = {'cws_file': cws_file,
                         'has_tissue': has_tissue}

            df = pd.DataFrame.from_dict(data_dict)
            df.to_csv(os.path.join(opts.results_dir, 'csv', sub_dir_name + '.csv'), index=False)

def save_detection_output_p(opts, sub_dir_name, image_path):
    make_sub_dirs(opts, sub_dir_name)
    
    files = sorted(glob.glob(os.path.join(opts.results_dir, 'mat', sub_dir_name, 'Ss1.mat')))

    for i in range(len(files)):
        mat_file_name = os.path.basename(files[i])
        # print('%s\n' % mat_file_name)
        image_path_full = os.path.join(image_path, mat_file_name[:-3] + 'jpg')
        save_detection_output(opts, sub_dir_name, mat_file_name, image_path_full)

def save_detection_output(opts, sub_dir_name, mat_file_name, image_path_full):
    mat = sio.loadmat(os.path.join(opts.results_dir, 'mat', sub_dir_name, mat_file_name))
    bin_label = np.argmax(np.array(mat['output']), axis=2)
    bin_label = np.ascontiguousarray(np.uint8(bin_label>0))
    strel = np.uint8(np.fromfunction(lambda x, y: (x-4)**2 + (y-4)**2 < 25, (9, 9), dtype=int))
    bin_label = cv2.dilate(bin_label, strel)
    _, contours, _ = cv2.findContours(bin_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bin_label = cv2.drawContours(bin_label, contours, -1, 1, cv2.FILLED)
    _, cc_label, stats, _ = cv2.connectedComponentsWithStats(bin_label)
    mat['BinLabel'] = (stats[cc_label, cv2.CC_STAT_AREA] >= 750) & (cc_label != 0)
    im = cv2.imread(image_path_full)
    annotated_image = (np.float32(im)/255.0)*np.tile(np.expand_dims(np.float32(mat['BinLabel']), axis=2), (1, 1, 3))
    if not opts.minimal_output:
        cv2.imwrite(os.path.join(opts.results_dir, 'annotated_images', sub_dir_name + '.png'), np.uint8(255.0*annotated_image))
    sio.savemat(os.path.join(opts.results_dir, 'mat', sub_dir_name, mat_file_name),  {'mat': mat})
    cv2.imwrite(os.path.join(opts.results_dir, 'mask', sub_dir_name + '_Mask.jpg'), cv2.cvtColor(np.uint8(255.0*annotated_image), cv2.COLOR_BGR2GRAY))

def generate_output(network, opts):
    cws_sub_dir = sorted(glob.glob(os.path.join(opts.data_dir, opts.file_name_pattern)))
    logits, _, _, _ = network.inference(images=network.images, is_training=False)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=opts.num_of_epoch)

    with tf.Session() as sess:

        ckpt = tf.train.get_checkpoint_state(opts.checkpoint_dir)
        assert ckpt, "No Checkpoint file found"
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Checkpoint file found at ' + ckpt.model_checkpoint_path)

        for cws_n in range(0, len(cws_sub_dir)):
            curr_cws_sub_dir = cws_sub_dir[cws_n]
            # print(curr_cws_sub_dir)
            sub_dir_name = os.path.basename(os.path.normpath(curr_cws_sub_dir))

            pre_process_images(opts=opts, sub_dir_name=sub_dir_name)
            try:
                generate_network_output(opts=opts, sub_dir_name=sub_dir_name, network=network, sess=sess, logits=logits)
            except:
                ss1 = cv2.imread(os.path.join(opts.data_dir, sub_dir_name, 'Ss1.jpg'))
                cv2.imwrite(os.path.join(opts.results_dir, 'mask', sub_dir_name + '_Mask.jpg'),cv2.cvtColor(ss1, cv2.COLOR_BGR2GRAY))

    return opts.results_dir


def generate_output_sub_dir(network, opts, sub_dir_name):
    logits, _, _, _ = network.inference(images=network.images, is_training=False)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=opts.num_of_epoch)

    with tf.Session() as sess:

        ckpt = tf.train.get_checkpoint_state(opts.checkpoint_dir)
        assert ckpt, "No Checkpoint file found"
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # print('Checkpoint file found at ' + ckpt.model_checkpoint_path)

        pre_process_images(opts=opts, sub_dir_name=sub_dir_name)
        try:
            generate_network_output(opts=opts, sub_dir_name=sub_dir_name, network=network, sess=sess, logits=logits)
        except:
            ss1 = cv2.imread(os.path.join(opts.data_dir, sub_dir_name, 'Ss1.jpg'))
            cv2.imwrite(os.path.join(opts.results_dir, 'mask', sub_dir_name + '_Mask.jpg'),cv2.cvtColor(ss1, cv2.COLOR_BGR2GRAY))
            
    return opts.results_dir
