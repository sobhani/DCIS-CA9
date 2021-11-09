import tensorflow as tf
import glob
import os
import numpy as np
import scipy.io as sio
import scipy.stats as sstats
import time
from datetime import datetime
import pandas as pd
import cv2

from step4_cell_class.subpackages import Patches
from step4_cell_class.subpackages import h5


def make_sub_dirs(opts, sub_dir_name):
    if not os.path.isdir(os.path.join(opts.results_dir, 'mat', sub_dir_name)):
        os.makedirs(os.path.join(opts.results_dir, 'mat', sub_dir_name))
    if not os.path.isdir(os.path.join(opts.results_dir, 'annotated_images', sub_dir_name)):
        os.makedirs(os.path.join(opts.results_dir, 'annotated_images', sub_dir_name))
    if not os.path.isdir(os.path.join(opts.results_dir, 'csv', sub_dir_name)):
        os.makedirs(os.path.join(opts.results_dir, 'csv', sub_dir_name))
    if not os.path.isdir(os.path.join(opts.preprocessed_dir, 'pre_processed', sub_dir_name)):
        os.makedirs(os.path.join(opts.preprocessed_dir, 'pre_processed', sub_dir_name))


def pre_process_images(opts, sub_dir_name):
    make_sub_dirs(opts, sub_dir_name)

    if opts.pre_process:
        if opts.tissue_segment_dir == '':
            files_tissue = sorted(glob.glob(os.path.join(opts.data_dir, sub_dir_name, 'Da*.jpg')))
        else:
            files_tissue = sorted(glob.glob(os.path.join(opts.tissue_segment_dir, 'mat', sub_dir_name, 'Da*.mat')))
                
        for i in range(len(files_tissue)):
            if not os.path.isfile(os.path.join(opts.preprocessed_dir, 'pre_processed', sub_dir_name, os.path.basename(files_tissue[i])[:-3]+'h5')):
                print('%s\n' % os.path.join(opts.preprocessed_dir, 'pre_processed', sub_dir_name, os.path.basename(files_tissue[i])[:-3]+'h5'))
                # target_image = np.float32(cv2.cvtColor(cv2.imread('Target.png'), cv2.COLOR_BGR2RGB))/255.0
                target_image = np.float32(cv2.cvtColor(cv2.imread('step4_cell_class/Target.png'), cv2.COLOR_BGR2RGB)) / 255.0
                image = np.float32(cv2.cvtColor(cv2.imread(os.path.join(opts.data_dir, sub_dir_name, os.path.basename(files_tissue[i])[:-3]+'jpg')), cv2.COLOR_BGR2RGB))/255.0

                if np.any(image):
                    image = norm_reinhard(image, target_image)
		
                feat = 255.0*image
                feat[feat < 0.0] = 0.0
                feat[feat > 255.0] = 255.0
                feat = np.round(feat)
            
                h5.h5write(os.path.join(opts.preprocessed_dir, 'pre_processed', sub_dir_name, os.path.basename(files_tissue[i])[:-3]+'h5'), feat, 'feat')
            else:
                print('Already Pre-Processed %s\n' % os.path.join(opts.preprocessed_dir, 'pre_processed', sub_dir_name, os.path.basename(files_tissue[i])[:-3]+'h5'))

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

def generate_network_output(opts, sub_dir_name, network, sess, logits_labels, csv_detection_results_dir):
    make_sub_dirs(opts, sub_dir_name)
    if opts.tissue_segment_dir == '':
        files_tissue = sorted(glob.glob(os.path.join(opts.data_dir, sub_dir_name, 'Da*.jpg')))
    else:
        files_tissue = sorted(glob.glob(os.path.join(opts.tissue_segment_dir, 'mat', sub_dir_name, 'Da*.mat')))

    for i in range(len(files_tissue)):
        file_name = os.path.basename(files_tissue[i])
        file_name = file_name[:-4]
        if not os.path.isfile(os.path.join(opts.results_dir, 'mat', sub_dir_name, file_name + '.mat')):
            print(file_name, flush=True)
            image_path_full = os.path.join(opts.data_dir, sub_dir_name, file_name + '.jpg')
            if opts.pre_process:
                feat = h5.h5read(
                    filename=os.path.join(opts.preprocessed_dir, 'pre_processed', sub_dir_name, file_name + '.h5'),
                    data_name='feat')
            else:
                feat = image_path_full

            patch_obj = Patches.Patches(patch_h=opts.image_height, patch_w=opts.image_width)

            image_patches, labels, cell_id = patch_obj.extract_patches(
                input_image=feat,
                input_csv=os.path.join(csv_detection_results_dir, file_name + '.csv'))
            opts.num_examples_per_epoch_train, opts.image_height, opts.image_width, opts.in_feat_dim = \
                image_patches.shape
            label_patches = np.zeros([opts.num_examples_per_epoch_train, opts.in_label_dim], dtype=np.float32)
            train_count = int((opts.num_examples_per_epoch_train / opts.batch_size) + 1)

            start = 0
            start_time = time.time()

            if image_patches.shape[0] != 0 and opts.batch_size > opts.num_examples_per_epoch_train:
                image_patches_temp = image_patches
                for rs_var in range(int((opts.batch_size / opts.num_examples_per_epoch_train))):
                    image_patches_temp = np.concatenate((image_patches_temp, image_patches), axis=0)

                image_patches = image_patches_temp

            opts.num_examples_per_epoch_train_temp = image_patches.shape[0]

            if image_patches.shape[0] != 0:
                label_patches = np.zeros([opts.num_examples_per_epoch_train_temp, opts.in_label_dim], dtype=np.float32)
                for step in range(train_count):
                    end = start + opts.batch_size
                    data_train = image_patches[start:end, :, :, :]
                    data_train = data_train.astype(np.float32, copy=False)
                    data_train_float32 = data_train / 255.0
                    logits_out = sess.run(
                        logits_labels,
                        feed_dict={network.images: data_train_float32,
                                   })
                    label_patches[start:end] = np.squeeze(logits_out, axis=1) + 1

                    if end + opts.batch_size > opts.num_examples_per_epoch_train_temp - 1:
                        end = opts.num_examples_per_epoch_train_temp - opts.batch_size

                    start = end

                label_patches = label_patches[0:opts.num_examples_per_epoch_train]
            duration = time.time() - start_time
            mat = {'output': label_patches,
                   'labels': labels,
                   'cell_ids': cell_id}
            sio.savemat(os.path.join(opts.results_dir, 'mat', sub_dir_name, file_name + '.mat'), mat)
            format_str = (
                '%s: file %d/ %d, (%.2f sec/file)')
            print(format_str % (datetime.now(), i + 1, len(files_tissue), float(duration)), flush=True)
        else:
            print('Already classified %s/%s\n' % (sub_dir_name, file_name), flush=True)


def post_process_images(opts, sub_dir_name, csv_detection_results_dir):
    files = sorted(glob.glob(os.path.join(opts.results_dir, 'mat', sub_dir_name, '*.mat')))
    
    for i in range(len(files)):
        mat_file_name = os.path.basename(files[i])[:-3]+'mat'
        csv_file_name = os.path.join(csv_detection_results_dir, os.path.basename(files[i])[:-3]+'csv')
        image_path_full = os.path.join(opts.data_dir, sub_dir_name, os.path.basename(files[i])[:-3]+'jpg')
        print('%s\n' % image_path_full)
        Save_Classification_Output(opts, sub_dir_name, mat_file_name, image_path_full, csv_file_name)

def Save_Classification_Output(opts, sub_dir_name, mat_file_name, image_path_full, csv_file_name):
    if not os.path.isfile(os.path.join(opts.results_dir, 'annotated_images', sub_dir_name, mat_file_name[:-3]+'png')):
        strength = 5
        A = pd.read_csv(csv_file_name)
        image = np.float64(cv2.cvtColor(cv2.imread(image_path_full), cv2.COLOR_BGR2RGB))/255.0
        cell_class = []
        colorcodes = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'colorcodes', opts.color_code_file))
        
        if not A.empty:
            detection = np.stack((np.array(A.loc[:, 'V2']), np.array(A.loc[:, 'V3'])), axis=1)
            mat = sio.loadmat(os.path.join(opts.results_dir, 'mat', sub_dir_name, mat_file_name))
            if 'mat' in mat:
                mat = mat['mat']
            output = np.array(mat['output'])-1
            cell_ids = np.reshape(np.array(mat['cell_ids']), output.shape)-1
            C = np.unique(cell_ids)
            cell_class = np.zeros(C.shape)
            for j in range(C.shape[0]):
                cell_class[j], _ = sstats.mode(output[cell_ids==C[j]])
            for c in range(len(colorcodes.index)):
                image = annotate_image_with_class(image, detection[cell_class==c,:], np.float64(hex2rgb(colorcodes.loc[c, 'color']))/255.0, strength)
            CC = colorcodes.loc[cell_class, 'class']
            CC.index=range(len(CC.index))
            A.loc[:, 'V1'] = CC
            A.to_csv(os.path.join(opts.results_dir, 'csv', sub_dir_name, mat_file_name[:-3]+'csv'), index=False)
        else:
            pd.DataFrame(data=None, columns=('V1', 'V2', 'V3')).to_csv(os.path.join(opts.results_dir, 'csv', sub_dir_name, mat_file_name[:-3]+'csv'))

        mat['class'] = cell_class
        cv2.imwrite(os.path.join(opts.results_dir, 'annotated_images', sub_dir_name, mat_file_name[:-3]+'png'), cv2.cvtColor(np.uint8(image*255.0), cv2.COLOR_RGB2BGR))
        sio.savemat(os.path.join(opts.results_dir, 'mat', sub_dir_name, mat_file_name),  {'mat': mat})
        print('saved %s\n' % os.path.join(opts.results_dir, 'annotated_images', sub_dir_name, mat_file_name[:-3]+'png'))
    else:
        print('Already saved %s\n' % os.path.join(opts.results_dir, 'annotated_images', sub_dir_name, mat_file_name[:-3]+'png'))
        
def hex2rgb(hx):
    hx = hx.lstrip('#')
    return np.array([int(hx[i:i+2], 16) for i in (0, 2, 4)])

def annotate_image_with_class(image, points, colour, strength):
    label = np.zeros(image.shape[0:2])
    label[points[:, 1]-1, points[:, 0]-1] = 1
    strel = np.uint8(np.fromfunction(lambda x, y: (x-strength+1)**2 + (y-strength+1)**2 < strength**2, ((2*strength)-1, (2*strength)-1), dtype=int))
    label = cv2.dilate(label, strel)>0
    image[label] = colour
    return image
    
def generate_output(network, opts, save_pre_process=True, network_output=True, post_process=True):
    cws_sub_dir = sorted(glob.glob(os.path.join(opts.data_dir, opts.file_name_pattern)))
    network.run_checks(opts=opts)
    logits, _ = network.inference(is_training=False)
    logits_labels = tf.argmax(logits[:, :, :, 0:network.num_of_classes], 3)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=opts.num_of_epoch)


    for cws_n in range(0, len(cws_sub_dir)):
        curr_cws_sub_dir = cws_sub_dir[cws_n]
        print(curr_cws_sub_dir, flush=True)
        sub_dir_name = os.path.basename(os.path.normpath(curr_cws_sub_dir))
        csv_detection_results_dir = os.path.join(opts.detection_results_path, 'csv', sub_dir_name)
        if save_pre_process:
            pre_process_images(opts=opts, sub_dir_name=sub_dir_name)

        if network_output:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(opts.checkpoint_dir)
                assert ckpt, "No Checkpoint file found"
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Checkpoint file found at ' + ckpt.model_checkpoint_path, flush=True)
                generate_network_output(opts=opts, sub_dir_name=sub_dir_name, network=network, sess=sess,
                                        logits_labels=logits_labels,
                                        csv_detection_results_dir=csv_detection_results_dir)

        if post_process:
            post_process_images(opts=opts, sub_dir_name=sub_dir_name,
                                csv_detection_results_dir=csv_detection_results_dir)

    return opts.results_dir


def generate_output_sub_dir(network, opts, sub_dir_name, save_pre_process=True, network_output=True, post_process=True):
    network.run_checks(opts=opts)
    logits, _ = network.inference(is_training=False)
    logits_labels = tf.argmax(logits[:, :, :, 0:network.num_of_classes], 3)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=opts.num_of_epoch)
    csv_detection_results_dir = os.path.join(opts.detection_results_path, 'csv', sub_dir_name)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(opts.checkpoint_dir)
        assert ckpt, "No Checkpoint file found"
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Checkpoint file found at ' + ckpt.model_checkpoint_path, flush=True)

        if save_pre_process:
            pre_process_images(opts=opts, sub_dir_name=sub_dir_name)

        if network_output:
            generate_network_output(opts=opts, sub_dir_name=sub_dir_name, network=network, sess=sess,
                                    logits_labels=logits_labels,
                                    csv_detection_results_dir=csv_detection_results_dir)

        if post_process:
            post_process_images(opts=opts, sub_dir_name=sub_dir_name,
                                csv_detection_results_dir=csv_detection_results_dir)

    return opts.results_dir
