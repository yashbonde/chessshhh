"""
this is the main file that is involved with
"""

import re
import chess
import numpy as np

BOARD_POS = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1', 'A2', 'B2', 'C2', 'D2', 'E2',
             'F2', 'G2', 'H2', 'A3', 'B3', 'C3', 'D3', 'E3', 'F3', 'G3', 'H3', 'A4', 'B4', 'C4', 'D4',
             'E4', 'F4', 'G4', 'H4', 'A5', 'B5', 'C5', 'D5', 'E5', 'F5', 'G5', 'H5', 'A6', 'B6', 'C6',
             'D6', 'E6', 'F6', 'G6', 'H6', 'A7', 'B7', 'C7', 'D7', 'E7', 'F7', 'G7', 'H7', 'A8', 'B8',
             'C8', 'D8', 'E8', 'F8', 'G8', 'H8']
FLIP_DICT = {"P": "P'", "N": "N'", "B": "B'", "R": "R'", "Q": "Q'", "K": "K'"}
FLIP_DICT2 = {"P'": 'p', "N'": 'n', "B'": 'b', "R'": 'r', "Q'": 'q', "K'": 'k'}


def flip_board_move(board, move):
    if not board.split()[1] == 'b':
        return board, move
    fen, _, castling_rights, enp, half_move, full_move = board.split()
    fen = fen[::-1].swapcase()
    if enp.upper() in BOARD_POS:
        enp = BOARD_POS[abs(63 - chess.__dict__[enp.upper()])].lower()
    castling_rights = board.split()[2]
    board = ' '.join([fen, 'w', castling_rights, enp, half_move, full_move])

    # now handle move
    move['from'] = abs(move['from'] - 63)
    move['to'] = abs(move['to'] - 63)
    board_ = chess.Board(board)
    del board_
    return board, move


def make_state(fen_string, return_legal=False):
    board = chess.Board(fen_string)
    state = np.zeros(64, np.uint8)
    for i in range(64):
        pp = board.piece_at(i)
        if pp is not None:
            state[i] = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
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
    state = state.reshape(8, 8)  # reshape the state to target

    # binary state
    b_state = np.zeros([4, 8, 8], np.uint8)
    b_state[0] = (state >> 3) & 1
    b_state[1] = (state >> 2) & 1
    b_state[2] = (state >> 1) & 1
    b_state[3] = (state >> 0) & 1

    if return_legal:
        return np.transpose(b_state, [1, 2, 0]).astype(np.uint8), list(board.legal_moves)
    return np.transpose(b_state, [1, 2, 0]).astype(np.uint8)


def match_best_legal_move(from_sq_probs_sorted, to_square_probs_sorted, legal_moves):
    for fsq, tsq in zip(from_sq_probs_sorted, to_square_probs_sorted):
        legal_common = list(filter(
            lambda move: (move.from_square, move.to_square) == (
                fsq, tsq), legal_moves
        ))
        if not legal_common:
            continue
        move = legal_common[0]
        break
    return move


def preprocess_states(bboards, bmoves, bresults):
    bstates = []
    bfrom = []
    bto = []
    bvalues = []
    for board, move_obj, res in zip(bboards, bmoves, bresults):
        if board.split()[1] == 'b':
            # this is the blacks move and we need to flip the board and fix the moves
            board, move_obj = flip_board_move(board, move_obj)
        state, _ = make_state(board)
        bstates.append(state)
        bfrom.append(move_obj['from'])
        bto.append(move_obj['to'])
        bvalues.append(res)

    bstates = np.array(bstates).astype(np.float32)
    bfrom = np.array(bfrom).astype(np.int32)
    bto = np.array(bto).astype(np.int32)
    bvalues = np.array(bvalues)

    return bstates, bfrom, bto, bvalues
