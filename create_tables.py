
from chess_engine.utils import md5_hash
from chess_engine.chess_db import games, moves, users
from chess_engine.chess_db.utils import connect

curr, conn = connect()
users.create_users_table(curr, conn)
games.create_games_table(curr, conn)
moves.create_moves_table(curr, conn)

def add_dummy_player(username, password = '1234'):

    # add few dummy robo players
    users.add_user(curr, conn, username, password, md5_hash(
        '{}--.--{}'.format(username, password)
    ))

for username in ['Arnold', 'Bernard', 'Robert Ford', 'Dolores']:
    add_dummy_player(username)

curr.close()
conn.close()
