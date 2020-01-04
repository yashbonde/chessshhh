"""this is the conventional data loader"""
# -*- coding: utf-8 -*-
# inspired from https://www.github.com/kyubyong/transformer

import tensorflow as tf
from .engine import make_state


def calc_num_batches(filenames, batch_size):
    total_lines = 0
    for f in filenames:
        for line in open(f):
            total_lines += 1
    total_lines -= 2 * len(filenames)
    return total_lines // batch_size + int(total_lines % batch_size != 0), total_lines


def generator_fn(filenames):
    for fname in filenames:
        for line_idx, line in enumerate(open(fname, 'r', encoding='utf-8')):
            if line_idx == 0:
                continue
            fen, fromsq, tosq, res = line.split(',')
            yield make_state(fen), int(fromsq), int(tosq), float(res)


def process_game(sample):
    sample = sample.decode('utf-8')
    game_state, from_sq, to_sq, result = sample.split(',')
    return game_state, int(from_sq), int(to_sq), int(result)


def input_fn(filenames, batch_size):
    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=((8, 8, 4), (), (), ()),
        output_types=(tf.uint8, tf.int32, tf.int32, tf.float32),
        args=(filenames,))  # <- arguments for generator_fn. converted to np string arrays
    dataset = dataset.repeat().shuffle(128*batch_size)
    dataset = dataset.padded_batch(
        batch_size, ((8, 8, 4), (), (), ())).prefetch(1)
    return dataset


def get_batch(filenames, batch_size):
    batches = input_fn(filenames=filenames, batch_size=batch_size)
    num_batches, total_lines = calc_num_batches(
        filenames=filenames, batch_size=batch_size)
    return batches, num_batches, total_lines


# def calc_num_batches(total_num, batch_size):
#     return total_num // batch_size + int(total_num % batch_size != 0)


# def load_data(fpath):
#     start_time = time.time()
#     game_strings = []
#     print(f'<><><><><><><><><><><> fpath: {fpath}')
#     all_lines = open(fpath, 'r', encoding='utf-8').readlines()
#     this_game = ''
#     for line_idx, line in enumerate(all_lines):
#         this_game += line
#         if line.split() and line.split()[-1] in ['0-1', '1-0', '1/2-1/2']:
#             game_strings.append(str(this_game))
#             this_game = ''
#     print(
#         f'-------------------- it took {time.time() - start_time}s to load the completely load the dataset')
#     print(f'--------------- Number of Samples: {len(game_strings)}')
#     print(f'--------------  return object type: {type(game_strings)}')
#     return list(game_strings)


# def generator_fn(all_games_list):
#     # all_games = load_data(file_path)
#     for game_string in all_games_list:
#         game = chess.pgn.read_game(io.StringIO(game_string.decode('utf-8')))
#         board = game.board()
#         game_result = RESULT_VALUE[game.headers['Result']]
#         for midx, move in enumerate(game.mainline_moves()):
#             # print(midx, board.san(move))
#             # print('--------------------------------------------move: {}'.format(midx))
#             move_obj = {'from': move.from_square, 'to': move.to_square}
#             # print('---> Board Input (Input)')
#             # print(board)
#             # print('---> Move:', move_obj, chess.Move(move_obj['from'], move_obj['to']))
#             board.push(chess.Move(move_obj['from'], move_obj['to']))
#             # print('---> Board before flipping (Actual)')
#             # print(board)
#             board_fen, move_obj = flip_board_move(board.fen(), move_obj)
#             # print('---> Board Post Flipping (State)')
#             # print(chess.Board(board_fen))
#             yield (make_state(board_fen)[0], move_obj['from'], move_obj['to'], game_result)


# def input_fn(all_games, batch_size, shuffle=False):
#     '''Returns:
#         tuple of (board [8x8X4], from_move, to_move, result)'''
#     shapes = ([8, 8, 4], [], [], [])
#     types = (tf.uint8, tf.int32, tf.int32, tf.float32)

#     print(
#         f'<><><><><><><><><> all_game info: {len(all_games)}, {type(all_games)}')
#     print(f'shapes: {shapes}')
#     print(f'types: {types}')

#     dataset = tf.data.Dataset.from_generator(
#         generator_fn,
#         output_shapes=shapes,
#         output_types=types,
#         args=((all_games,))
#     )

#     dataset = dataset.repeat()  # iterate forever
#     if shuffle:  # for training
#         dataset.shuffle(32*batch_size)
#     dataset = dataset.padded_batch(batch_size, shapes).prefetch(1)
#     return dataset


# def get_batch(fpath, batch_size, shuffle=False):
#     all_games = load_data(fpath=fpath)
#     batches = input_fn(all_games=all_games,
#                        batch_size=batch_size, shuffle=shuffle)
#     num_batches = calc_num_batches(
#         total_num=len(all_games), batch_size=batch_size)
#     # return batches
#     return batches, num_batches
