from .data_loader import get_batch
from .net import network
from chess.pgn import read_game
from tqdm import tqdm
import chess
import tensorflow as tf
import numpy as np
import time
import os
import io
import argparse
from chess_engine.zima.trainer import train_network_supervised
from glob import glob

parser = argparse.ArgumentParser(description='train chess player')
parser.add_argument('--name', type=str, default='pokemon',
                    help='name of the model')
parser.add_argument('--games_data', type =str, default = './games_data')
config = parser.parse_args()

"""This is the main trainer file for our model with all the utils and
main trainer network and functions"""
RESULT_VALUE = {'0-1': -1., '1/2-1/2': 0., '1-0': 1.}

def train_network_supervised(filepaths, log_folder):
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
                # print('>>> _from: {}\n_to: {}\n_value: {}\n_from_loss: {}\n_to_loss:'
                #  '{}\nvalue_loss: {}\ntotal_loss: {}'.format(
                #      np.argmax(_from, axis = 1), np.argmax(_to, axis = 1), _value, _from_loss, _to_loss,
                #      _value_loss, _from_loss + _to_loss + _value_loss
                #  ))
