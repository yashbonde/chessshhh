"""utility functions"""

import os
import json
import chess
import logging
import hashlib
import argparse

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


class ModelConfig:
    """
    Custom config handler. It has following schema
    ModelConfig
    |---> description
    |---> config (all command line arguments)
    |---> extern (externally added key value pairs, path to checkpoint and models)
    """

    def __init__(self, description='', path=None, loading=False):
        self.flag_json = {
            'description': description,
            'config': {},
            'extern': {}
        }
        self.description = description
        self.path = path

        if loading:
            self.load_from_json()
        else:
            self.ap = argparse.ArgumentParser(description=description)

    def add_arg(self, flag, type, default='', help='', **kwargs):
        """
        Add CLI argument
        :param flag: name/flag
        :param type: dtype
        :param default: default value if any
        :param help: help string
        :param kwargs: kwargs sent to `arparse.ArgumentParser().add_argument()` method
        """
        self.ap.add_argument(flag, default=default,
                             type=type, help=help, **kwargs)
        self.flag_json['config'][flag[2:]] = None

    def add_value(self, flag, value):
        setattr(self, flag, value)
        self.flag_json['config'][flag] = value

    def add_key_value(self, flag, value):
        """
        Add simple flag-value argument to configuration, added to `extern` sub object
        """
        self.flag_json['extern'][flag] = value
        setattr(self, flag, value)

    def parse_args(self):
        """
        parse command line args
        """
        self.ap = self.ap.parse_args()

        for flag in self.flag_json['config']:
            val = getattr(self.ap, flag)
            setattr(self, flag, val)
            self.flag_json['config'][flag] = val

        del self.ap  # save memory

    def save_json(self):
        """
        Save config json to path
        """
        config_str = json.dumps(self.flag_json)
        logging.info("Model Config:\n{}".format(config_str.replace(',', '\n')))
        logging.info("Saving model configuration at {}".format(self.path))
        with open(self.path, 'w', encoding='utf-8') as f:
            f.write(config_str)

    def load_from_json(self):
        """
        load config from json
        """
        logging.warning(
            "Loading model configuration from {}".format(self.path))

        res = json.load(open(self.path))
        self.flag_json = res

        self.description = self.flag_json['description']
        for k, v in self.flag_json['config'].items():
            setattr(self, k, v)
        for k, v in self.flag_json['extern'].items():
            setattr(self, k, v)
