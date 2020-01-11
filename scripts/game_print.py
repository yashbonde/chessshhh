import chess
import json
import io
import time
from chess.pgn import read_game

RESULT_VALUE = {'0-1': -1, '1/2-1/2':0, '1-0': 1}

def load_data(fpath):
    start_time = time.time()
    game_strings = []
    print(f'<><><><><><><><><><><> fpath: {fpath}')
    all_lines = open(fpath, 'r', encoding='utf-8').readlines()
    this_game = ''
    for line_idx, line in enumerate(all_lines):
        this_game += line
        if line.split() and line.split()[-1] in ['0-1', '1-0', '1/2-1/2']:
            game_strings.append(str(this_game))
            this_game = ''
    print('-------------------- it took {:.3f}s to load the completely load the dataset'.format(time.time() - start_time))
    print(f'--------------- Number of Samples: {len(game_strings)}')
    return list(game_strings)

def generator_fn(all_games_list):
    # all_games = load_data(file_path)
    for game_string in all_games_list:
        try:
            game = chess.pgn.read_game(io.StringIO(game_string.decode('utf-8')))
        except:
            game = chess.pgn.read_game(io.StringIO(game_string))
        board = game.board()
        game_result = RESULT_VALUE[game.headers['Result']]
        for midx, move in enumerate(game.mainline_moves()):
            print('---> Board Input (pre): {}'.format(board.fen()))
            print(board)
            print('---> Move:', move)
            board.push(move)
            print('---> Board after move (post): {}'.format(board.fen()))
            print(board)
            print('\n\n')

generator_fn(load_data('../KB_small.pgn'))