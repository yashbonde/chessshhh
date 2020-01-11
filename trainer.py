from chess.pgn import read_game
from tqdm import tqdm
import chess
import tensorflow as tf
import numpy as np
import time
import os
import io
import argparse
from glob import glob
from types import SimpleNamespace

def train_alphazero_network(config):
    if not config.value_alphazero:
        raise ValueError("Network is not alphazero configured")
    from chess_engine.zima_value import value_net, value_utils

    filepaths = glob(config.glob_expression)
    print('----- filepaths: {}\n'.format('\n'.join(filepaths)))
    batches, _, _ = value_utils.get_batches(filepaths, config.batch_size,
        config.gram_size, config.shuffle)
    tf_iterator = tf.data.Iterator.from_structure(batches.output_types, batches.output_shapes)
    xs, ys = tf_iterator.get_next()
    train_init_op = tf_iterator.make_initializer(batches)
    print('\n\n##########, xs, ys: {}, {}\n\n'.format(xs, ys))

    iter_obj = SimpleNamespace(
        board = xs,
        value_target=ys
    )
    zima_net = value_net.value_network_alphazero(iter_obj, config)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_init_op)
        train_writer = tf.summary.FileWriter(config.log_folder, sess.graph)
        saver_ = tf.train.Saver(max_to_keep=config.max_saves)
        save_path = config.log_folder + '/' + config.name + '.ckpt'
        global_step = 0

        for step in tqdm(range(config.num_steps)):
            ops = [zima_net.train_step, zima_net.loss, zima_net.summary]
            _, _l, _sum = sess.run(ops)
            global_step += 1

            if step and step % config.log_every == 0:
                train_writer.add_summary(_sum, global_step)

            if step and step % config.save_every == 0:
                saver_.save(sess, save_path)


def train_network_supervised_policy(filepaths, log_folder):
    from chess_engine.zima_common.data_loader import get_batch
    from chess_engine.zima_policy.net import network

    # main code
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    batches, num_batches, tot_samples = get_batch(filepaths, batch_size=config.batch_size)
    iter = tf.data.Iterator.from_structure(
        batches.output_types, batches.output_shapes)
    xs = iter.get_next()
    train_init_op = iter.make_initializer(batches)
    zima_net = network(xs, training=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_init_op)
        log_file = open(log_folder + '/logs.txt', 'a')
        save_path = log_folder + '/' + log_folder.split('/')[-1] + '.ckpt'
        train_writer = tf.summary.FileWriter(log_folder, sess.graph)
        saver_ = tf.train.Saver(max_to_keep=config.max_saves)
        global_step = 0
        for epoch_idx in range(config.num_epochs):
            ops = [
                zima_net.pred_from,
                zima_net.pred_to,
                zima_net.value,
                zima_net.loss_from,
                zima_net.loss_to,
                zima_net.loss_values,
                zima_net.train_step,
                zima_net.summary
            ]
            for _ in tqdm(range(num_batches)):
                _from, _to, _value, _from_loss, _to_loss, _value_loss, _, _sum = sess.run(
                    ops)
                train_writer.add_summary(_sum, global_step)
                global_step += 1
                if epoch_idx == 0:
                    saver_.save(sess, save_path)
                
if __name__ == "__main__":
    from chess_engine.utils import ModelConfig

    config = ModelConfig('trainer for some shitty ass chess network')
    config.add_arg('--m', str, 'alphazero_value', 'name of model either of "alphazero_value"')
    config.add_arg('--name', str, 'bubble', 'name or id of this version')
    config.add_arg('--lr', float, 0.2, 'learning rate')
    config.add_arg('--reg_const', float, 0.002, 'regularised loss factor')
    config.add_arg('--batch_size', int, 32, 'minibatch size')
    config.add_arg('--gram_size', int, 8, 'windo size into past used for training')
    config.add_arg('--glob_expression', str, 'data/*_full.pgn', 'glob epression for games')
    config.add_arg('--max_saves', int, 3, 'maximum copies to save')
    config.add_arg('--num_steps', int, 100, 'num of training steps')
    config.add_arg('--log_every', int, 10, 'log every this steps (includes tensroboard logs)')
    config.add_arg('--save_every', int, 400, 'save model every')
    config.add_arg('--seed', int, 4, 'seed value for randomness')
    config.parse_args()
    config.add_key_value('lr_drops', [10, 30, 50])
    config.add_key_value('training', True)
    config.add_key_value('value_alphazero', True)
    config.add_key_value('shuffle', True)
    config.add_key_value('log_folder', config.name + '/')
    if not os.path.exists(config.log_folder):
        os.makedirs(config.log_folder)
    config.path = os.path.join(config.log_folder, 'model_config.json')
    config.save_json()

    # feed seed values
    tf.random.set_random_seed(config.seed)
    np.random.seed(config.seed)

    if config.m == 'alphazero_value':
        train_alphazero_network(config)


