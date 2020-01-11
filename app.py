"""main runner file for our chess engine setup
13.11.2019 - @yashbonde"""

import uuid
import random
import logging
from flask import request, make_response
from flask import Flask, jsonify

# custom
from chess_engine.utils import md5_hash, get_move_obj, get_board_state
from chess_engine.ai_move import move_orchestrator
from chess_engine.chess_db import moves, games, users
from chess_engine.chess_db.utils import connect

# make the app and run the server
app = Flask(__name__)

CURR, CONN = connect()
AI_PLAYERS = ['Arnold', 'Bernard', 'Robert Ford', 'Dolores Abarnathy']
STARTING_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'


@app.route("/")
def index():
  return open("index.html").read()


@app.route('/login', methods = ["POST"])
def login():
    payload = request.get_json()

    username = payload.get("username")
    password = payload.get("password")

    # get authenticated
    selection = users.get_users(CURR, username, password)
    if not selection:
        return make_response(jsonify(
            auth_token = None
        ))

    # now we set auth token
    auth_token = md5_hash("{user_rev}.{password}.{uuid_str}".format(
        user_rev = username[::-1],
        password = password,
        uuid_str = str(uuid.uuid4())
    ))
    users.update_auth_token(CURR, CONN, username, auth_token) # update auth token
    users.update_session(CURR, CONN, auth_token) # update session
    return make_response(jsonify(
        auth_token = auth_token
    ))


@app.route('/new_user', methods = ["POST"])
def new_user():
    payload = request.get_json()

    username = payload.get("username")
    password = payload.get("password")

    player_hash = md5_hash("{user_rev}--.--{password}".format(
        user_rev=username[::-1],
        password=password))

    users.add_user(CURR, CONN, username, password, player_hash)

    return make_response(jsonify(
        message = "user {} created".format(username)
    ))


@app.route('/new_game', methods = ["GET"])
def new_game():

    payload = request.headers
    auth_token = payload.get("Authentication")
    print(auth_token)

    # assign any random opponent
    opp_name = random.choice(AI_PLAYERS)
    auth_token_opp = md5_hash("{user_rev}.{password}.{uuid_str}".format(
        user_rev=opp_name[::-1],
        password='1234',
        uuid_str=str(uuid.uuid4())
    )) # now we set auth token
    users.update_auth_token(CURR, CONN, opp_name, auth_token_opp)  # update auth token

    # make new game with this opponent
    games.add_game(CURR, CONN, auth_token, auth_token_opp) # make new game
    game_id = games.get_last_game(CURR)[0]['id']
    # moves.add_move_using_auth(CURR, CONN, game_id, auth_token, 0, STARTING_FEN, 'START')

    # TODO: Add functionality for player 2 i.e when user opts to be black

    # reset for the new game
    return make_response(jsonify(
        board_state = STARTING_FEN,
        game_id = game_id,
        player_no = 0 # will be 0 for white and 1 for black
    ))


@app.route('/move', methods = ["POST"])
def make_move():
    # get payload data
    payload = request.get_json()
    auth_token = payload["auth_token"]
    game_id = payload['game_id']
    from_ = payload['from']
    to_ = payload['target']
    player_no = payload['player_no']
    board_fen = payload['board_fen']
    san = payload['san'] # this is the notation we would like to save}

    # update move in DB for current player and game state if required
    moves.add_move_using_auth(CURR, CONN, game_id, auth_token, board_fen, san)
    if san[-1] == '#': # this meanst that the game has ended
        games.end_game(CURR, CONN, game_id)
        return make_response(jsonify(
            board_state = board_fen,
            from_square = None,
            to_square = None,
            content = 'Checkmate, You Won! Proceed to a new game, my child!'
        ))

    # feed the AI new state and get legal moves
    res = move_orchestrator(board_fen)

    # update the move for the opponent plus update the game state if needed
    moves.add_opp_move_using_auth(CURR, CONN, player_no, game_id, auth_token, res['new_state'], res['san'])
    if res['content'] is not None:
        games.end_game(CURR, CONN, game_id)
        games.end_game(CURR, CONN, game_id)
        return make_response(jsonify(
            board_state = board_fen,
            from_square = None,
            to_square = None,
            content = res['content']
        ))

    # return the new move
    return make_response(jsonify(
        board_state = res['new_state'],
        from_square = res['from'].lower(),
        to_square = res['to'].lower(),
        content = res['content']
    ))


@app.route('/start_game', methods = ["GET"])
def start_game():
    # get player hash
    auth_token = request.headers.get("Authentication")
    res = users.get_player_hash_for_auth_token(CURR, auth_token)['player_hash']

    # plug game manager here

    # return okay status
    return make_response(jsonify(
        content = "go"
    ))

if __name__ == "__main__":
    app.run(debug=True)
