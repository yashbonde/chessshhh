"""
this is the main trainer file for value networks. This network
takes in the n-grammed states from games and returns probability of
victory in the game as if it was the one playing the game.
"""

import logging
from types import SimpleNamespace
import tensorflow as tf
from chess_engine.zima_common import common_layers

ACTIVATION = {
    "relu": tf.nn.relu,
}


def get_lr_strategy(config, global_step):
    lr_init = config.lr
    lr_drops = config.lr_drops
    lrs = [lr_init/10**(drop + 1) for drop in range(len(lr_drops))]
    lr = tf.constant(lr_init)
    for lr_updated, lr_step in zip(lrs, lr_drops):
        lr = tf.cond(global_step > lr_step,
                     lambda: tf.constant(lr_updated), lambda: lr)
    return lr


def value_network_alphazero(iter_obj, config, lr=0.2):
    """
    This is only the value head part for this network and not the policy head
    piece, so I think we may get away with using less number of layers. The
    alpha zero network is defined as (p,v) = f_theta(s) whereas this only
    value network is defined as v = f_theta(s) where theta are the parameters

    As far as the training is concerned the learning rate was dropped twice
    during the training. It starts with 0.2, then is dropped to 0.02 at 100k,
    then to 0.002 at 300k and 0.0002 at 500k.
    """
    board = tf.cast(iter_obj.board, tf.float32)
    batch_size = common_layers.shapes_list(board)[0]
    # defined is the regularisation loss
    with tf.variable_scope("zima_alphazero_value",
        regularizer=tf.keras.regularizers.l2()):

        # first projection
        out = tf.layers.batch_normalization(
            tf.layers.conv2d(board, 256, 3, padding='same'),
            training=config.training)
        for layer_idx in range(19):
            with tf.variable_scope("stack_{}".format(layer_idx)):
                conv_out = tf.nn.relu(
                    tf.layers.batch_normalization(
                        tf.layers.conv2d(out, 256, 3, padding='same'),
                        training=config.training
                    ))
                conv_out = tf.nn.relu(
                    tf.layers.batch_normalization(
                        tf.layers.conv2d(conv_out, 256, 3, padding='same'),
                        training=config.training
                    ))
                out += conv_out

        out = tf.nn.relu(
            tf.layers.batch_normalization(
                tf.layers.conv2d(out, 1, 1, padding='same'),
                training=config.training
            ))
        out = tf.reshape(out, [batch_size, 64], name='flatten')
        out = tf.nn.relu(tf.layers.dense(out, 265))
        value_out = tf.nn.tanh(tf.layers.dense(out, 1))

    if config.training:
        # we need to define the loss function here
        """
        loss function is defined as follows:
        l = (z-v)^2 + c||theta||^2

        where ||x|| is L2 normalisation of x
        """
        value_target = tf.expand_dims(
            tf.cast(iter_obj.value_target, tf.float32),
            axis=1)
        value_loss = tf.reduce_sum(
            tf.math.squared_difference(value_out, value_target))
        reg_losses = config.reg_const * sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = value_loss + reg_losses

        # adding to summaries
        tf.summary.scalar("value_loss", value_loss)
        tf.summary.scalar("reg_losses", reg_losses)
        tf.summary.scalar("total_loss", total_loss)

        # making a trainstep
        global_step = tf.train.get_or_create_global_step()
        lr = get_lr_strategy(config, global_step)
        tf.summary.scalar("learning rate", lr)
        train_step = tf.train.GradientDescentOptimizer(lr).minimize(total_loss)
        summary = tf.summary.merge_all()

        return SimpleNamespace(
            train_step = train_step,
            loss = total_loss,
            summary=summary,
            global_step=global_step
        )
    
    return SimpleNamespace(
        value_out=value_out
    )


def value_network(iter_obj, config, lr=0.001, log_eps=0.0001):
    # this is the main network func
    board = tf.cast(iter_obj.board, tf.float32)
    value_target = tf.expand_dims(
        tf.cast(iter_obj.value_target, tf.float32),
        axis=1)

    print(f'board: {board}\nvalue_: {value_target}')

    with tf.variable_scope('zima_value'):
        out = tf.layers.conv2d(board, config.filters[0], config.kernel_size[0],
            config.strides[0], padding = 'same')

        for layer_idx in range(config.conv_layers):
            with tf.variable_scope('conv_{}'.format(layer_idx)):
                new_out = tf.layers.conv2d(out,
                    config.filters[layer_idx],
                    config.kernel_size[layer_idx],
                    config.strides[layer_idx],
                    padding = 'same')
                new_out = common_layers.normalize_tensor(
                    new_out,
                    type = config.norm_type[layer_idx],
                    is_training=config.training
                )
                new_out = ACTIVATION[config.activations[layer_idx]](new_out)
                if layer_idx and layer_idx % config.residual_every == 0:
                    out += new_out
                else:
                    out = new_out
        
        out = tf.layers.flatten(out)

        for layer_idx in range(config.dense_layers):
            with tf.variable_scope('dense_{}'.format(layer_idx)):
                out = tf.layers.dense(
                    out,
                    config.dense_features[layer_idx],
                    ACTIVATION[config.dense_activations[layer_idx]]
                )
                out = common_layers.normalize_tensor(out,
                    config.dense_norm_types[layer_idx],
                    is_training=config.training)
                
        value_pred = tf.nn.tanh(tf.layers.dense(out, 1, name = 'value_head'))
        # can there be a mthod to represent probability curves to near accuracy and
        # then identify the curve and make neural network predict the variables!

    if config.training:
        # value loss
        # loss_values = tf.reduce_sum(tf.sqrt(
        #     tf.abs(tf.pow(value, 2) - tf.pow(value_placeholder), 2))
        # )) # sqrt(abs(v_pred^2 - v^2))
        loss = tf.abs(- 0.5 * tf.reduce_sum(
            (1 - value_target) * tf.log(tf.abs(1 - value_pred + config.log_eps)) +
            (1 + value_target) * tf.log(tf.abs(1 + value_pred + config.log_eps))
        ))

        # total_loss and train steps
        train_step = tf.train.AdamOptimizer(config.lr).minimize(loss)

        # # accuracy
        # equal = tf.math.equal(tf.argmax(value_pred))
        # acc = tf.math.reduce_sum(
        #     tf.cast(tf.equal, tf.float32)
        # )

        # tf.summary.scalar('value_accuracy', (tf.reduce_mean(value_pred) + 1)/2)
        tf.summary.scalar('loss', loss)
        summary = tf.summary.merge_all()

        return SimpleNamespace(
            value=value_pred,
            loss=loss,
            train_step=train_step,
            summary=summary
        )

    return SimpleNamespace(
        value=value_pred,
        board = iter_obj.board
    )
