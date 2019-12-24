"""This script converts the games in all the different file to a
single csv file. Then use tf.data.csv_reader"""

import os
import re
import io
import time
import chess
import numpy as np
from chess import pgn
from glob import glob
from tqdm import tqdm

board_pos = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1', 'A2', 'B2', 'C2', 'D2', 'E2',
    'F2', 'G2', 'H2', 'A3', 'B3', 'C3', 'D3', 'E3', 'F3', 'G3', 'H3', 'A4', 'B4', 'C4', 'D4',
    'E4', 'F4', 'G4', 'H4', 'A5', 'B5', 'C5', 'D5', 'E5', 'F5', 'G5', 'H5', 'A6', 'B6', 'C6',
    'D6', 'E6', 'F6', 'G6', 'H6', 'A7', 'B7', 'C7', 'D7', 'E7', 'F7', 'G7', 'H7', 'A8', 'B8',
    'C8', 'D8', 'E8', 'F8', 'G8', 'H8']
RESULT_VALUE = {'0-1': -1., '1/2-1/2': 0., '1-0': 1.}
FLIP_DICT = {"P": "P'", "N": "N'", "B": "B'", "R": "R'", "Q": "Q'", "K": "K'"}
FLIP_DICT2 = {"P'": 'p', "N'": 'n', "B'": 'b', "R'": 'r', "Q'": 'q', "K'": 'k'}

# -- functions -- #
def flip_board_move_b2w(board, move):
    if not board.split()[1] == 'b':
        return board, move
    fen, _, castling_rights, enp, half_move, full_move = board.split()
    fen = ''.join(reversed(fen))
    if enp.upper() in board_pos:
        enp = board_pos[abs(63 - chess.__dict__[enp.upper()])].lower()
    castling_rights = board.split()[2]
    for k,v in FLIP_DICT.items():
        fen = re.sub(k, v, fen)
    fen = fen.upper()
    for k, v in FLIP_DICT2.items():
        fen = re.sub(k, v, fen)
    board = ' '.join([fen, 'w', castling_rights, enp, half_move, full_move])
    # now handle move
    move['from'] = abs(move['from'] - 63)
    move['to'] = abs(move['to'] - 63)
    board_ = chess.Board(board)
    del board_
    return board, move

def load_data(fpath):
    print('Processing file ---> {}'.format(fpath))
    start_time = time.time()
    game_strings =[]
    all_lines = open(fpath, 'r', encoding = 'latin').readlines()
    this_game = ''
    for line_idx, line in enumerate(all_lines):
        this_game += line
        if line.split() and line.split()[-1] in ['0-1', '1-0', '1/2-1/2']:
            game_strings.append(str(this_game))
            this_game = ''
    print('---> it took {}s to load the dataset'.format(time.time() - start_time))
    print('---> Number of Samples: {}'.format(len(game_strings)))
    return list(game_strings)

# --- Script --- #
master_file = '../games_data/master_{}_{}.csv'
for fpath in sorted(['KingBase2019-A00-A39.pgn'
    ,'KingBase2019-A40-A79.pgn'
    ,'KingBase2019-A80-A99.pgn'
    ,'KingBase2019-B00-B19.pgn'
    ,'KingBase2019-B20-B49.pgn'
    ,'KingBase2019-B50-B99.pgn'
    ,'KingBase2019-C00-C19.pgn'
    ,'KingBase2019-C20-C59.pgn'
    ,'KingBase2019-C60-C99.pgn'
    ,'KingBase2019-D00-D29.pgn'
    ,'KingBase2019-D30-D69.pgn'
    ,'KingBase2019-D70-D99.pgn'
    ,'KingBase2019-E00-E19.pgn'
    ,'KingBase2019-E20-E59.pgn'
    ,'KingBase2019-E60-E99.pgn']):
    start_time = time.time()
    sub_child = 0
    this_file_name = master_file.format(fpath.split('.')[0], sub_child)
    if os.path.exists(this_file_name):
        print('File: {} exists. Skipping'.format(this_file_name))
        continue
    try:
        this_file_strings = load_data('../games_data/{}'.format(fpath))
    except Exception as e:
        print(e)
        continue
    open(this_file_name, 'a', encoding='utf-8').write('board,from,to,result')
    for idx in tqdm(range(len(this_file_strings))):
        try:
            game_string = this_file_strings[idx]
            game = pgn.read_game(io.StringIO(game_string))
            board = game.board()
            game_result = RESULT_VALUE[game.headers['Result']]
            write_strings = [] # making it a batch process for faster processing
            for midx, move in enumerate(game.mainline_moves()):
                move_obj = {'from': move.from_square, 'to': move.to_square}
                board.push(chess.Move(move_obj['from'], move_obj['to']))
                board_fen, move_obj = flip_board_move_b2w(board.fen(), move_obj)
                write_strings.append('{},{},{},{}'.format(
                    board_fen, move_obj['from'], move_obj['to'], game_result
                ))
            open(this_file_name, 'a', encoding='utf-8').write('\n'+'\n'.join(write_strings))
            fstat = os.stat(this_file_name)
            if fstat.st_size // (8 * 1024) > 300:
                sub_child += 1
                print('Making new subchild file, file became larger than 300MB')

        except Exception as e:
            print('Error:', e)
    print('-------- done with file (took: {}s) ------'.format(time.time() - start_time))