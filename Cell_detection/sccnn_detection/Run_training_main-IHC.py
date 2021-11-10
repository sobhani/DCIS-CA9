import os
from shutil import copyfile
import pickle
import scipy.io as sio
import sccnn_detection as sccnn
from subpackages import NetworkOptions


opts = NetworkOptions.NetworkOptions(exp_dir=os.path.join(os.getcwd(), 'ExpDir-IHC'),
                                     num_examples_per_epoch_train=1,
                                     num_examples_per_epoch_valid=1,
                                     image_height=31,
                                     image_width=31,
                                     in_feat_dim=3,
                                     label_height=13,
                                     label_width=13,
                                     in_label_dim=1,
                                     batch_size=1000,
                                     num_of_epoch=500,
                                     data_dir=r'D:\tmp',
                                     train_data_filename='TTF1-TrainData180517',
                                     valid_data_filename='TTF1-ValidData180517',
                                     current_epoch_num=0)

if not os.path.isdir(opts.exp_dir):
    os.makedirs(opts.exp_dir)
    os.makedirs(opts.checkpoint_dir)
    os.makedirs(opts.log_train_dir)
    os.makedirs(os.path.join(opts.exp_dir, 'subpackages'))

copyfile('Run_training_main-IHC.py', os.path.join(opts.exp_dir, 'Run_training_main-IHC.py'))
copyfile('sccnn_detection.py', os.path.join(opts.exp_dir, 'sccnn_detection.py'))
files = os.listdir(os.path.join(os.getcwd(), 'subpackages'))
for file in files:
    if file.endswith('.py'):
        copyfile(os.path.join(os.getcwd(), 'subpackages', file),
                 os.path.join(opts.exp_dir, 'subpackages', file))

mat = {'opts': opts}
sio.savemat(os.path.join(opts.exp_dir, 'opts.mat'), mat)
pickle.dump(opts, open(os.path.join(opts.exp_dir, 'opts.p'), 'wb'))


Network = sccnn.SCCNN(batch_size=opts.batch_size,
                      image_height=opts.image_height,
                      image_width=opts.image_width,
                      in_feat_dim=opts.in_feat_dim,
                      out_height=opts.label_height,
                      out_width=opts.label_width,
                      out_feat_dim=opts.in_label_dim,
                      radius=opts.maxclique_distance)
Network = Network.run_training(opts=opts)

