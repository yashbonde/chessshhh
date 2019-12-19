"""utility functions"""

import os
import chess
import hashlib

DB_STRING = os.environ["DB_PATH"].format(user=os.environ["POSTGRES_USER"],
                                         password=os.environ["POSTGRES_PASSWORD"],
                                         host=os.environ["POSTGRES_HOST"],
                                         db_name=os.environ["POSTGRES_DB"])

board_position_to_int = {'A1': 0,
                         'B1': 1,
                         'C1': 2,
                         'D1': 3,
                         'E1': 4,
                         'F1': 5,
                         'G1': 6,
                         'H1': 7,
                         'A2': 8,
                         'B2': 9,
                         'C2': 10,
                         'D2': 11,
                         'E2': 12,
                         'F2': 13,
                         'G2': 14,
                         'H2': 15,
                         'A3': 16,
                         'B3': 17,
                         'C3': 18,
                         'D3': 19,
                         'E3': 20,
                         'F3': 21,
                         'G3': 22,
                         'H3': 23,
                         'A4': 24,
                         'B4': 25,
                         'C4': 26,
                         'D4': 27,
                         'E4': 28,
                         'F4': 29,
                         'G4': 30,
                         'H4': 31,
                         'A5': 32,
                         'B5': 33,
                         'C5': 34,
                         'D5': 35,
                         'E5': 36,
                         'F5': 37,
                         'G5': 38,
                         'H5': 39,
                         'A6': 40,
                         'B6': 41,
                         'C6': 42,
                         'D6': 43,
                         'E6': 44,
                         'F6': 45,
                         'G6': 46,
                         'H6': 47,
                         'A7': 48,
                         'B7': 49,
                         'C7': 50,
                         'D7': 51,
                         'E7': 52,
                         'F7': 53,
                         'G7': 54,
                         'H7': 55,
                         'A8': 56,
                         'B8': 57,
                         'C8': 58,
                         'D8': 59,
                         'E8': 60,
                         'F8': 61,
                         'G8': 62,
                         'H8': 63}

int_to_board_position = {v:k for k,v in board_position_to_int.items()}

def get_move_obj(source, target):
    move = chess.Move(
        from_square=board_position_to_int[source.upper()],
        to_square=board_position_to_int[target.upper()]
    )
    return move


def get_board_state(pre_board_state_fen, move):
    board = chess.Board(pre_board_state_fen)
    try:
        board.push(move)
    except:
        return None
    return board.fen()


def md5_hash(string):
    return hashlib.md5(string.encode("utf-8")).hexdigest()
