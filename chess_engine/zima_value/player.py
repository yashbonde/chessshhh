"""this is the mail file that has functions to play with the network

MIG 676, Pallavi Vihar Sector-2, Gol Chouk, DDU-Nagar, Rohinipuram, Raipur"""

import os
import json
import chess
import numpy as np
import tensorflow as tf
from types import SimpleNamespace
from collections import OrderedDict

from chess_engine import utils
from chess_engine.zima_common import engine
from chess_engine.zima_value.value_net import value_network_alphazero

class ValuePlayerNetworkWrapper:
    def __init__(self):
        # paths and config
        environ_path = os.environ["ALPHAZERO_MODEL_NAME"]
        model_config_path = environ_path + '/model_config.json'
        model_save_path = environ_path + '/' + environ_path + '.ckpt'
        print('environ_path: {}'.format(environ_path))
        print('model_config_path: {}'.format(model_config_path))
        print('model_save_path: {}'.format(model_save_path))

        config = utils.ModelConfig(path = model_config_path, loading=True)
        config.training = False
        self.config = config

        # network and initialisation
        print('Making and loading network from {}'.format(model_save_path))
        self.tf_board = SimpleNamespace(
            board=tf.placeholder(
                tf.uint8, [None, 8, 8, config.gram_size * 5], name='board')
        )
        self.net = value_network_alphazero(self.tf_board, config)
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, model_save_path)

        # make state function
        osmodel_name = os.environ["MODEL_NAME"]
        if osmodel_name == "alphazero_value":
            self.state_fn = self._make_state_alphazero_value

    def _make_state_alphazero_value(self, fens):
        gram_size = self.config.gram_size
        game_states = [engine.make_state(game, player_layer=True) for game in fens[::-1]]
        target_state = game_states[0]
        for game in game_states[1:]:
            target_state = np.append(target_state, game, axis=-1)
        for _ in range(gram_size - len(fens)):
            target_state = np.append(
                    target_state, np.zeros(shape=(8, 8, 5)), axis=-1)
        return target_state

    def get_values(self, board_fens):
        state = self.state_fn(board_fens)
        return self.sess.run(self.net.value, {
            self.tf_board.board: state
        })

    def run_one_step_greedy(self, board_fens):
        moves, boards = [], []
        board = chess.Board(board_fens[-1])
        for move in self.board.legal_moves:
            self.board.push(move)
            board_fen = self.board.fen()
            boards.append(board_fen)
            moves.append(move)
            self.board.pop()
        print('------> boards: {}'.format(boards))
        value_of_states = self.get_values(boards)
        move_to_make = moves[np.argmax(value_of_states)]
        self.board.push(move_to_make)
        self.boards.append(self.board.fen())
        return self.board.fen(), value_of_states, moves

