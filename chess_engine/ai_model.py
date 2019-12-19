"""this is the super smart AI model that can play any chess"""

import chess
import random

class Turk(object):
    def __init__(self):
        pass

    # def make_random_move(self, board_state_fen, opp_move):
    #     board = chess.board(board_state_fen)
    #     board.push(opp_move)
    #     legal_moves = list(board.legal_moves)
    #     move = random.choice(legal_moves)
    #     board.push(move)
    #     return move.uci(), board.fen()

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