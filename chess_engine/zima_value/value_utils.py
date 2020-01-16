# this has the function that are necessary for training the value network.

import io
import json
import chess
import collections
import numpy as np
import tensorflow as tf
from chess_engine.zima_common import engine


def load_data(fpath, ret_len=False):
    game_strings = []
    all_lines = open(fpath, 'r', encoding='utf-8').readlines()
    this_game = ''
    for line_idx, line in enumerate(all_lines):
        this_game += line
        if line.split() and line.split()[-1] in ['0-1', '1-0', '1/2-1/2']:
            game_strings.append(str(this_game))
            this_game = ''
    if ret_len:
        return len(game_strings)
    return list(game_strings)


def generator_fn(all_games, gram_size=3):
    for game_string in all_games:
        states_buffer = collections.deque()
        try:
            game = chess.pgn.read_game(
                io.StringIO(game_string.decode('utf-8')))
        except:
            game = chess.pgn.read_game(io.StringIO(game_string))
        board = game.board()
        game_result = engine.RESULT_VALUE[game.headers['Result']]
        player_old = 1
        for midx, move in enumerate(game.mainline_moves()):
            if midx == 0:
                states_buffer.appendleft(board.fen())
            if len(states_buffer) == gram_size:
                # return game state as is
                games = list(states_buffer)
                game_states = [engine.make_state(
                    game, player_layer=True) for game in games]
                target_state = game_states[0]
                for game in game_states[1:]:
                    target_state = np.append(target_state, game, axis=-1)
                yield target_state, int(game_result)
                states_buffer.pop()

            elif len(states_buffer) < gram_size:
                # stack with zeros is the idea
                games = list(states_buffer)
                game_states = [engine.make_state(
                    game, player_layer=True) for game in games]
                target_state = game_states[0]
                for game in game_states[1:]:
                    target_state = np.append(target_state, game, axis=-1)
                for _ in range(gram_size - len(states_buffer)):
                    target_state = np.append(
                        target_state, np.zeros(shape=(8, 8, 5)), axis=-1)
                yield target_state, int(game_result)

            # once the current board is returned update the move
            board.push(move)
            states_buffer.appendleft(board.fen())
            game_result *= -1.


def input_fn(all_games, batch_size, gram_size, shuffle=False):
    '''Returns:
        tuple of (board [stack_size, 8x8X4], from_move, to_move, result)'''
    shapes = ([8, 8, 5 * gram_size], [])
    types = (tf.uint8, tf.int32)

    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(all_games, gram_size,)
    )

    dataset = dataset.repeat()  # iterate forever
    if shuffle:  # for training
        dataset.shuffle(128*batch_size)
    dataset = dataset.padded_batch(batch_size, shapes).prefetch(1)
    return dataset


def get_batches(file_paths, batch_size, gram_size, shuffle=False):
    tot_games = 0
    all_games = []
    for fpath in file_paths:
        tot_games += load_data(fpath, True)
        all_games.extend(load_data(fpath))
    approx_total_moves = tot_games * 50
    batches = input_fn(all_games, batch_size, gram_size, shuffle)
    return batches, approx_total_moves // batch_size + int(approx_total_moves % batch_size != 0), approx_total_moves
