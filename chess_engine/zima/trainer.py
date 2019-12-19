"""This is the main trainer file for our model with all the utils and
main trainer network and functions"""

# import dependencies
import io
import os
import time
import numpy as np
import tensorflow as tf
import chess
from tqdm import tqdm
from chess.pgn import read_game

# zima dependencies
from engine import make_state, preprocess_states, flip_board_move
from net import network

RESULT_VALUE = {'0-1': -1., '1/2-1/2': 0., '1-0': 1.}
import tensorflow as tf

def calc_num_batches(total_num, batch_size):
    return total_num // batch_size + int(total_num % batch_size != 0)

def load_data(fpath):
    start_time = time.time()
    game_strings =[]
    print(f'<><><><><><><><><><><> fpath: {fpath}')
    all_lines = open(fpath, 'r', encoding = 'utf-8').readlines()
    this_game = ''
    for line_idx, line in enumerate(all_lines):
        this_game += line
        if line.split() and line.split()[-1] in ['0-1', '1-0', '1/2-1/2']:
            game_strings.append(str(this_game))
            this_game = ''
    print(f'-------------------- it took {time.time() - start_time}s to load the completely load the dataset')
    print(f'--------------- Number of Samples: {len(game_strings)}')
    print(f'--------------  return object type: {type(game_strings)}')
    return list(game_strings)

def generator_fn(all_games_list):
    # all_games = load_data(file_path)
    for game_string in all_games_list:
        game = chess.pgn.read_game(io.StringIO(game_string.decode('utf-8')))
        board = game.board()
        game_result = RESULT_VALUE[game.headers['Result']]
        for midx, move in enumerate(game.mainline_moves()):
            # print(midx, board.san(move))
            # print('--------------------------------------------move: {}'.format(midx))
            move_obj = {'from': move.from_square, 'to': move.to_square}
            # print('---> Board Input (Input)')
            # print(board)
            # print('---> Move:', move_obj, chess.Move(move_obj['from'], move_obj['to']))
            board.push(chess.Move(move_obj['from'], move_obj['to']))
            # print('---> Board before flipping (Actual)')
            # print(board)
            board_fen, move_obj = flip_board_move(board.fen(), move_obj)
            # print('---> Board Post Flipping (State)')
            # print(chess.Board(board_fen))
            yield (make_state(board_fen)[0], move_obj['from'], move_obj['to'], game_result)

def input_fn(all_games, batch_size, shuffle=False):
    '''Returns:
        tuple of (board [8x8X4], from_move, to_move, result)'''
    shapes = ([8,8,4], [], [], [])
    types = (tf.uint8, tf.int32, tf.int32, tf.float32)

    print(f'<><><><><><><><><> all_game info: {len(all_games)}, {type(all_games)}')
    print(f'shapes: {shapes}')
    print(f'types: {types}')

    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=((all_games,))
    )

    dataset = dataset.repeat() # iterate forever
    if shuffle: # for training
        dataset.shuffle(32*batch_size)
    dataset = dataset.padded_batch(batch_size, shapes).prefetch(1)
    return dataset

def get_batch(fpath, batch_size, shuffle=False):
    all_games = load_data(fpath = fpath)
    batches = input_fn(all_games = all_games, batch_size = batch_size, shuffle=shuffle)
    num_batches = calc_num_batches(total_num=len(all_games), batch_size=batch_size)
    # return batches
    return batches, num_batches

def train_network_supervised(file_path, log_folder, num_epochs = 400, batch_size = 64, max_saves = 4):
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    batches, num_batches = get_batch(fpath = file_path, batch_size = 3, shuffle = True)
    iter = tf.data.Iterator.from_structure(
        batches.output_types, batches.output_shapes)
    xs = iter.get_next()
    train_init_op = iter.make_initializer(batches)  
    print(f'=============== xs: {xs}')
    zima_net = network(xs, training = True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_init_op)
        log_file = open(log_folder + '/logs.txt', 'a')
        save_path = log_folder + '/' + log_folder.split('/')[-1] + '.ckpt'
        train_writer = tf.summary.FileWriter(log_folder, sess.graph)
        saver_ = tf.train.Saver(max_to_keep = max_saves)
        global_step = 0
        for epoch_idx in range(num_epochs):
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
                _from, _to, _value, _from_loss, _to_loss, _value_loss, _, _sum = sess.run(ops)
                train_writer.add_summary(_sum, global_step)
                global_step += 1
                if epoch_idx == 0:
                    saver_.save(sess, save_path)
                # print('>>> _from: {}\n_to: {}\n_value: {}\n_from_loss: {}\n_to_loss:'
                #  '{}\nvalue_loss: {}\ntotal_loss: {}'.format(
                #      np.argmax(_from, axis = 1), np.argmax(_to, axis = 1), _value, _from_loss, _to_loss,
                #      _value_loss, _from_loss + _to_loss + _value_loss
                #  ))
        