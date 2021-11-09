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
import multiprocessing
from functools import partial
from step4_cell_class.subpackages.cellPos import merge_csv_files
from step4_cell_class.subpackages.cell_count import cell_count

from step4_cell_class.subpackages import Patches


def make_sub_dirs(opts, sub_dir_name):
    if not os.path.isdir(os.path.join(opts.results_dir, 'CellScore')):
        os.makedirs(os.path.join(opts.results_dir, 'CellScore'))
    if not os.path.isdir(os.path.join(opts.results_dir, 'CellPos')):
        os.makedirs(os.path.join(opts.results_dir, 'CellPos'))
    if not os.path.isdir(os.path.join(opts.results_dir, 'mat', sub_dir_name)):
        os.makedirs(os.path.join(opts.results_dir, 'mat', sub_dir_name))
    if (not opts.minimal_output) and (not os.path.isdir(os.path.join(opts.results_dir, 'annotated_images', sub_dir_name))):
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
         
        with multiprocessing.Pool(max(1, multiprocessing.cpu_count()-1)) as pool:
            pool.map(partial(pre_process_images_par, opts=opts, sub_dir_name=sub_dir_name), files_tissue)
                   

def pre_process_images_par(file_path, opts, sub_dir_name):
    if not os.path.isfile(os.path.join(opts.preprocessed_dir, 'pre_processed', sub_dir_name, os.path.basename(file_path)[:-3]+'mat')):
        # print('%s\n' % os.path.join(opts.preprocessed_dir, 'pre_processed', sub_dir_name, os.path.basename(file_path)[:-3]+'mat'))
        # target_image = np.float32(cv2.cvtColor(cv2.imread('step4_cell_class/Target.png'), cv2.COLOR_BGR2RGB))/255.0
        target_image = np.float32(cv2.cvtColor(cv2.imread(os.path.join(os.path.dirname(__file__), '..', 'Target.png')), cv2.COLOR_BGR2RGB))/255.0
        image = np.float32(cv2.cvtColor(cv2.imread(os.path.join(opts.data_dir, sub_dir_name, os.path.basename(file_path)[:-3]+'jpg')), cv2.COLOR_BGR2RGB))/255.0

        if np.any(image):
            image = norm_reinhard(image, target_image)
		
        feat = 255.0*image
        feat[feat < 0.0] = 0.0
        feat[feat > 255.0] = 255.0
        feat = np.round(feat)
            
        sio.savemat(os.path.join(opts.preprocessed_dir, 'pre_processed', sub_dir_name, os.path.basename(file_path)[:-3] + 'mat'), {'matlab_output': {'feat': feat}})
    else:
        # print('Already Pre-Processed %s\n' % os.path.join(opts.preprocessed_dir, 'pre_processed', sub_dir_name, os.path.basename(file_path)[:-3] + 'mat'))
        pass


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
            # print(file_name, flush=True)
            image_path_full = os.path.join(opts.data_dir, sub_dir_name, file_name + '.jpg')
            if opts.pre_process:
                workspace = sio.loadmat(os.path.join(opts.preprocessed_dir, 'pre_processed', sub_dir_name,
                                                     file_name + '.mat'))
                # workspace = sio.loadmat(os.path.join(opts.results_dir, 'pre_processed', sub_dir_name,
                #                                      file_name + '.mat'))
                matlab_output = workspace['matlab_output']
                feat = np.array(matlab_output['feat'][0][0])
            else:
                feat = image_path_full

            patch_obj = Patches.Patches(patch_h=opts.image_height, patch_w=opts.image_width)

            image_patches, labels, cell_id = patch_obj.extract_patches(
                input_image=feat,
                input_csv=os.path.join(csv_detection_results_dir, file_name + '.csv'))
            image_patches = image_patches.astype(np.float32, copy=False)/255.0
        
            opts.num_examples_per_epoch_train, opts.image_height, opts.image_width, opts.in_feat_dim = \
                image_patches.shape
            label_patches = np.zeros([opts.num_examples_per_epoch_train, opts.in_label_dim], dtype=np.float32)

            data_train = np.zeros((opts.batch_size, opts.image_height, opts.image_width, opts.in_feat_dim), dtype=np.float32)
        
            start_time = time.time()

            for start in range(0, opts.num_examples_per_epoch_train, opts.batch_size):
                end = min(start + opts.batch_size, opts.num_examples_per_epoch_train)
                
                data_train[:(end-start), :, :, :] = image_patches[start:end, :, :, :]
                
                logits_out = sess.run(
                    logits_labels,
                    feed_dict={network.images: data_train,
                               })
                label_patches[start:end, :] = np.squeeze(logits_out[:(end-start), :], axis=1) + 1
                
            duration = time.time() - start_time
            mat = {'output': label_patches,
                   'labels': labels,
                   'cell_ids': cell_id}
            sio.savemat(os.path.join(opts.results_dir, 'mat', sub_dir_name, file_name + '.mat'), mat)
            format_str = (
                '%s: file %d/ %d, (%.2f sec/file)')
            # print(format_str % (datetime.now(), i + 1, len(files_tissue), float(duration)), flush=True)
        else:
            # print('Already classified %s/%s\n' % (sub_dir_name, file_name), flush=True)
            pass


def post_process_images(opts, sub_dir_name, csv_detection_results_dir, color_code_path, cell_pos_path, cell_score_path):
    files = sorted(glob.glob(os.path.join(opts.results_dir, 'mat', sub_dir_name, '*.mat')))

    with multiprocessing.Pool(max(1, multiprocessing.cpu_count()-1)) as pool:
        pool.map(partial(Save_Classification_Output, opts=opts, sub_dir_name=sub_dir_name, csv_detection_results_dir=csv_detection_results_dir, color_code_path=color_code_path), files)
        
    if not os.path.isfile(cell_pos_path):
        merge_csv_files(os.path.join(opts.data_dir, sub_dir_name), os.path.join(opts.results_dir, 'csv', sub_dir_name), cell_pos_path)
    
    if not os.path.isfile(cell_score_path):
        cell_classes = tuple(pd.read_csv(color_code_path).loc[:, 'class'])
        cell_count(cell_pos_path, cell_score_path, sub_dir_name, classes=cell_classes)

def Save_Classification_Output(file_path, opts, sub_dir_name, csv_detection_results_dir, color_code_path):
    mat_file_name = os.path.basename(file_path)[:-3]+'mat'
    csv_file_name = os.path.join(csv_detection_results_dir, os.path.basename(file_path)[:-3]+'csv')
    
    image_path_full = os.path.join(opts.data_dir, sub_dir_name, os.path.basename(file_path)[:-3]+'jpg')
    # print('%s\n' % image_path_full)
    
    if not os.path.isfile(os.path.join(opts.results_dir, 'annotated_images', sub_dir_name, mat_file_name[:-3]+'png')):
        strength = 5
        A = pd.read_csv(csv_file_name)
        cell_class = []
        colorcodes = pd.read_csv(color_code_path)
        
        mat = sio.loadmat(os.path.join(opts.results_dir, 'mat', sub_dir_name, mat_file_name))
        if 'mat' in mat:
            mat = mat['mat']
                
        if not A.empty:
            detection = np.stack((np.array(A.loc[:, 'V2']), np.array(A.loc[:, 'V3'])), axis=1)
            output = np.array(mat['output'])-1
            cell_ids = np.reshape(np.array(mat['cell_ids']), output.shape)-1
            C = np.unique(cell_ids)
            cell_class = np.zeros(C.shape)
            for j in range(C.shape[0]):
                cell_class[j], _ = sstats.mode(output[cell_ids==C[j]])
            CC = colorcodes.loc[cell_class, 'class']
            CC.index=range(len(CC.index))
            A.loc[:, 'V1'] = CC
            A.to_csv(os.path.join(opts.results_dir, 'csv', sub_dir_name, mat_file_name[:-3]+'csv'), index=False)
        else:
            pd.DataFrame(data=None, columns=('V1', 'V2', 'V3')).to_csv(os.path.join(opts.results_dir, 'csv', sub_dir_name, mat_file_name[:-3]+'csv'), index=False)

        if not opts.minimal_output:
            mat['class'] = cell_class
            sio.savemat(os.path.join(opts.results_dir, 'mat', sub_dir_name, mat_file_name),  {'mat': mat})
            
            image = np.float64(cv2.cvtColor(cv2.imread(image_path_full), cv2.COLOR_BGR2RGB))/255.0
            if not A.empty:
                for c in range(len(colorcodes.index)):
                    image = annotate_image_with_class(image, detection[cell_class==c,:], np.float64(hex2rgb(colorcodes.loc[c, 'color']))/255.0, strength)
            cv2.imwrite(os.path.join(opts.results_dir, 'annotated_images', sub_dir_name, mat_file_name[:-3]+'png'), cv2.cvtColor(np.uint8(image*255.0), cv2.COLOR_RGB2BGR))
        # print('saved %s\n' % os.path.join(opts.results_dir, 'annotated_images', sub_dir_name, mat_file_name[:-3]+'png'))
    else:
        # print('Already saved %s\n' % os.path.join(opts.results_dir, 'annotated_images', sub_dir_name, mat_file_name[:-3]+'png'))
        pass

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
        # print(curr_cws_sub_dir, flush=True)
        sub_dir_name = os.path.basename(os.path.normpath(curr_cws_sub_dir))
        
        csv_detection_results_dir = os.path.join(opts.detection_results_path, 'csv', sub_dir_name)
        color_code_path = os.path.join(os.path.dirname(__file__), '..', 'colorcodes', opts.color_code_file)
        cell_pos_path = os.path.join(opts.results_dir, 'CellPos', sub_dir_name + '_cellPos.csv')
        cell_score_path = os.path.join(opts.results_dir, 'CellScore', sub_dir_name + '_cellScore.csv')
        
        if save_pre_process:
            pre_process_images(opts=opts, sub_dir_name=sub_dir_name)

        if network_output:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(opts.checkpoint_dir)
                assert ckpt, "No Checkpoint file found"
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                # print('Checkpoint file found at ' + ckpt.model_checkpoint_path, flush=True)
                generate_network_output(opts=opts, sub_dir_name=sub_dir_name, network=network, sess=sess,
                                        logits_labels=logits_labels,
                                        csv_detection_results_dir=csv_detection_results_dir)

        if post_process:
            post_process_images(opts=opts, sub_dir_name=sub_dir_name,
                                csv_detection_results_dir=csv_detection_results_dir, color_code_path=color_code_path, cell_pos_path=cell_pos_path, cell_score_path=cell_score_path)

    return opts.results_dir


def generate_output_sub_dir(network, opts, sub_dir_name, save_pre_process=True, network_output=True, post_process=True):
    network.run_checks(opts=opts)
    logits, _ = network.inference(is_training=False)
    logits_labels = tf.argmax(logits[:, :, :, 0:network.num_of_classes], 3)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=opts.num_of_epoch)
    
    csv_detection_results_dir = os.path.join(opts.detection_results_path, 'csv', sub_dir_name)
    color_code_path = os.path.join(os.path.dirname(__file__), '..', 'colorcodes', opts.color_code_file)
    cell_pos_path = os.path.join(opts.results_dir, 'CellPos', sub_dir_name + '_cellPos.csv')
    cell_score_path = os.path.join(opts.results_dir, 'CellScore', sub_dir_name + '_cellScore.csv')

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(opts.checkpoint_dir)
        assert ckpt, "No Checkpoint file found"
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # print('Checkpoint file found at ' + ckpt.model_checkpoint_path, flush=True)

        if save_pre_process:
            pre_process_images(opts=opts, sub_dir_name=sub_dir_name)

        if network_output:
            generate_network_output(opts=opts, sub_dir_name=sub_dir_name, network=network, sess=sess,
                                    logits_labels=logits_labels,
                                    csv_detection_results_dir=csv_detection_results_dir)

        if post_process:
            post_process_images(opts=opts, sub_dir_name=sub_dir_name,
                                csv_detection_results_dir=csv_detection_results_dir, color_code_path=color_code_path, cell_pos_path=cell_pos_path, cell_score_path=cell_score_path)

    return opts.results_dir
