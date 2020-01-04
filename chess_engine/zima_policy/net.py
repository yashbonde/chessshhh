"""
This is the main network file for ZIMA

8.12.2019 - @yashbonde
"""

import logging
import tensorflow as tf
import numpy as np
from types import SimpleNamespace
from chess_engine.zima.common_layers

def network(iter_obj, block_size = 4, training = False, lr = 0.001, log_eps = 0.0001):
    # this is the main network func
    board = tf.cast(iter_obj[0], tf.float32)
    from_action = tf.cast(iter_obj[1], tf.int32)
    to_action = tf.cast(iter_obj[2], tf.int32)
    value_target = tf.cast(iter_obj[3], tf.float32)
    value_ = tf.expand_dims(value_target, axis = 1)

    print(f'----------\n\
    board: {board}\nfrom_action: {from_action}\nto_action: {to_action}\
    \nvalue_: {value_}\n-----------')

    with tf.variable_scope('zima'):
        # a tonne of conv layers
        filter_size = 32
        kernel_size = 3
        stride = (1,1)
        out = tf.layers.conv2d(board, filter_size, kernel_size, stride, padding = 'same')
        residual = None
        logging.info('start [block_size]: {}, {}'.format(block_size, out))
        # for layer_idx in range(block_size * 3):
        #     with tf.variable_scope('conv_{}'.format(layer_idx)):
        #         if layer_idx and layer_idx % block_size == 0: # time to drop
        #             stride = (2,2)
        #             filter_size *= 2
        #             kernel_size -= 1
        #         if layer_idx and layer_idx % 2 == 0 and residual is not None: # add residual block
        #             out += residual
        #             logging.info('>>> RESIDUAL')
        #         out = tf.nn.relu(tf.layers.conv2d(out, filter_size, kernel_size, stride, padding = 'same'))
        #         out = normalise_tensor(out, 'norm')
        #         residual = out
        #         logging.info('layer: {}; k: {}; tensor: {}'.format(layer_idx, kernel_size, out))
        #         stride = (1,1)
        out = tf.nn.relu(tf.layers.conv2d(board, 128, (2,2), padding = 'same'))
        out = tf.layers.dropout(out, 0.1, training = True)
        out = tf.nn.relu(tf.layers.conv2d(out, 128, (2,2), (2,2), padding = 'same'))
        out = tf.layers.flatten(out)
        out = tf.layers.dropout(out, 0.1, training = True)
        out = tf.layers.dense(out, 1024, activation = tf.nn.relu)
        out = tf.layers.dense(out, 1024)
        # out += residual
        logging.info('output from conv blocks [+ Residual]: {}'.format(out))
        
        # # now we flatten tensor and go for ff layers
        # out_shape = shapes_list(out)
        # out = tf.reshape(out, [out_shape[0], out_shape[1] * out_shape[2] * out_shape[3]], name = 'flatten')
        # game_emb = tf.nn.relu(ff(out, 128, 'ff_emb'))
        
        # from things
        pred_from_logits = ff(out, 64, 'ff_from')
        pred_from = tf.nn.softmax(pred_from_logits, name = 'from_square')

        # to things
        pred_to_logits = ff(out, 64, 'ff_to')
        pred_to = tf.nn.softmax(pred_to_logits, name = 'to_square')

        # value network
        out = tf.nn.relu(ff(out, 32, 'ff_value_0'))
        value_pred = tf.nn.tanh(ff(out, 1, 'value'))

        print('------\npred_from: {}\npred_to: {}\nvalue_pred: {}\npred_from_logits: {}\npred_to_logits: {}\n------'.format(
            pred_from, pred_to, value_pred, pred_from_logits, pred_to_logits))

    if training:
        # loss for source box
        loss_from = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels = tf.one_hot(from_action, 64),
                logits = pred_from_logits
            )
        )

        # loss for to box
        loss_to = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels = tf.one_hot(to_action, 64),
                logits = pred_to_logits
            )
        )

        # value loss
        # loss_values = tf.reduce_sum(tf.sqrt(
        #     tf.abs(tf.pow(value, 2) - tf.pow(value_placeholder), 2))
        # )) # sqrt(abs(v_pred^2 - v^2))
        loss_values = tf.abs(- 0.5 * tf.reduce_sum(
            (1 - value_) * tf.log(tf.abs(1 - value_pred + log_eps)) + 
            (1 + value_) * tf.log(tf.abs(1 + value_pred + log_eps))
        ))

        # total_loss and train steps
        total_loss = loss_from + loss_to + loss_values
        train_step = tf.train.AdamOptimizer(lr).minimize(total_loss)

        tf.summary.scalar('to_loss', loss_to)
        tf.summary.scalar('from_loss', loss_from)
        tf.summary.scalar('value', (tf.reduce_mean(value_pred) + 1)/2)
        tf.summary.scalar('total_loss', total_loss)
        summary = tf.summary.merge_all()

        return SimpleNamespace(
            value = tf.squeeze(value_pred),
            pred_from = pred_from,
            pred_to = pred_to,
            loss_from = loss_from,
            loss_to = loss_to,
            loss_values = loss_values,
            train_step = train_step,
            summary = summary
        )
    
    return SimpleNamespace(
        value = value_pred,
        pred_from = pred_from,
        pred_to = pred_to
    )
