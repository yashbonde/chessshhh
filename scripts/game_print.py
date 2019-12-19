import re
import io
import time
import chess
import numpy as np
from chess import pgn
import sentencepiece as spm

board_pos = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1', 'A2', 'B2', 'C2', 'D2', 'E2',
    'F2', 'G2', 'H2', 'A3', 'B3', 'C3', 'D3', 'E3', 'F3', 'G3', 'H3', 'A4', 'B4', 'C4', 'D4',
    'E4', 'F4', 'G4', 'H4', 'A5', 'B5', 'C5', 'D5', 'E5', 'F5', 'G5', 'H5', 'A6', 'B6', 'C6',
    'D6', 'E6', 'F6', 'G6', 'H6', 'A7', 'B7', 'C7', 'D7', 'E7', 'F7', 'G7', 'H7', 'A8', 'B8',
    'C8', 'D8', 'E8', 'F8', 'G8', 'H8']
RESULT_VALUE = {'0-1': -1., '1/2-1/2': 0., '1-0': 1.}
FLIP_DICT = {"P": "P'", "N": "N'", "B": "B'", "R": "R'", "Q": "Q'", "K": "K'"}
FLIP_DICT2 = {"P'": 'p', "N'": 'n', "B'": 'b', "R'": 'r', "Q'": 'q', "K'": 'k'}

# -- functions -- #
def flip_board_move(board, move):
    # print(board)
    # print('----> In move:', move)
    # handle for board
    if not board.split()[1] == 'b':
        print('++++++++++++++++ No flipping needed')
        return board, move
    print('++++++++++++++++ Post flipping')
    fen, _, castling_rights, enp, half_move, full_move = board.split()
    fen = ''.join(reversed(fen))
    # print(fen, enp)
    if enp.upper() in board_pos:
        print(f'....... enp: {enp}')
        enp = board_pos[abs(63 - chess.__dict__[enp.upper()])].lower()
        print(f'.....>> enp: {enp}')
    castling_rights = board.split()[2]
    for k,v in FLIP_DICT.items():
        fen = re.sub(k, v, fen)
        # castling_rights = re.sub(k, v, castling_rights)
    fen = fen.upper()
    # castling_rights = castling_rights.upper()
    for k, v in FLIP_DICT2.items():
        fen = re.sub(k, v, fen)
        # castling_rights = re.sub(k, v, castling_rights)
    board = ' '.join([fen, 'w', castling_rights, enp, half_move, full_move])
    
    print('----> Board Post Flipping (FEN):', board)

    # now handle move
    move['from'] = abs(move['from'] - 63)
    move['to'] = abs(move['to'] - 63)
    board_ = chess.Board(board)
    # print(board_)
    print('---> Move [F]:', move, chess.Move(move['from'], move['to']))
    del board_
    return board, move

def load_data(fpath):
    start_time = time.time()
    game_strings =[]
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

def make_state(fen_string):
    board = chess.Board(fen_string)
    state = np.zeros(64, np.uint8)
    for i in range(64):
        pp = board.piece_at(i)
        if pp is not None:
            state[i] = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,\
                "p": 9, "n": 10, "b": 11, "r": 12, "q": 13, "k": 14}[pp.symbol()]
    castling_rights = fen_string.split()[2]
    if board.has_queenside_castling_rights(chess.WHITE):
        assert state[0] == 4
        state[0] = 7
    if board.has_kingside_castling_rights(chess.WHITE):
        assert state[7] == 4
        state[7] = 7
    if board.has_queenside_castling_rights(chess.BLACK):
        assert state[56] == 12
        state[56] = 8 + 7
    if board.has_kingside_castling_rights(chess.BLACK):
        assert state[63] == 12
        state[63] = 8 + 7

    if board.ep_square is not None:
        assert state[board.ep_square] == 0
        state[board.ep_square] = 8
    state = state.reshape(8, 8) # reshape the state to target

    # binary state
    b_state = np.zeros([4, 8, 8], np.uint8)
    b_state[0] = (state>>3)&1
    b_state[1] = (state>>2)&1
    b_state[2] = (state>>1)&1
    b_state[3] = (state>>0)&1

    return np.transpose(b_state, [1, 2, 0]).astype(np.uint8), list(board.legal_moves)

for game_string in load_data('KB_small.pgn'):
    print('\n============================ NEW GAME =============================\n')
    game = pgn.read_game(io.StringIO(game_string))
    board = game.board()
    game_result = RESULT_VALUE[game.headers['Result']]
    for midx, move in enumerate(game.mainline_moves()):
        # print(midx, board.san(move))
        print('--------------------------------------------move: {}'.format(midx))
        move_obj = {'from': move.from_square, 'to': move.to_square}
        print('---> Board Input (Input):', board.fen())
        print(board)
        print('---> Move:', move_obj, chess.Move(move_obj['from'], move_obj['to']))
        board.push(chess.Move(move_obj['from'], move_obj['to']))
        print('---> Board before flipping (Actual):', board.fen())
        print(board)
        board_fen, move_obj = flip_board_move(board.fen(), move_obj)
        print('---> Board Post Flipping (State)', board_fen)
        print(chess.Board(board_fen))
        state, _ = make_state(board_fen)
        print('---> State: {}'.format(state.shape))