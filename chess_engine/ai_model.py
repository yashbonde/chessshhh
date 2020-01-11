"""this is the super smart AI model that can play any chess"""

import os
import chess
import random

from chess_engine.zima_value

def make_random_move(board_state_fen):
    board = chess.Board(board_state_fen)

    if board.is_checkmate():
        return "checkmate", board.fen(), None
    elif board.is_stalemate():
        return "stalemate", board.fen(), None

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return "end", board.fen()
    move = random.choice(legal_moves)
    san = board.san(move)
    board.push(move)
    return move, board.fen(), san