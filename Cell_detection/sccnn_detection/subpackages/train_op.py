import tensorflow as tf


def train(network, loss=None):
    if loss is None:
        loss = network.loss
    with tf.name_scope('Optimization'):
        # Optimization
        train_step_unet = tf.train.AdagradOptimizer(learning_rate=network.LearningRate).minimize(loss)
        network.train_op = train_step_unet

    return network.train_op
