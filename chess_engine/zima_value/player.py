"""this is the mail file that has functions to play with the network"""
import os
import chess
import tensorflow as tf
from types import SimpleNamespace
from collections import OrderedDict

def run_one_step(board, move, sess, config, network):
    moves = []
    boards = []
    for move in board.legal_moves():
        board.move(move)
        board_fen = board.fen()
        boards.append(board_fen)
        moves.append(move)
        boards.pop()
    make_states = [make_state(v) for b in boards]
    value_of_states = sess.run(network.values, {network.boards : make_states})
    move_to_make = moves[np.argmax(value_of_states)]
    board.move(move_to_make)
    return board, value_of_states, moves

def start_game():
