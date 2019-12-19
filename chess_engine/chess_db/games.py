from chess_engine.chess_db.utils import execute

def create_games_table(cursor, conn):
    create_table_games = '''CREATE TABLE games (
	id SERIAL PRIMARY KEY NOT NULL,
	created time DEFAULT CURRENT_TIMESTAMP,
	ended time,
	deleted time,
	player_one char(32) NOT NULL,
	player_two char(32) NOT NULL,
	status varchar (100),
	ruleset varchar (20),
	FOREIGN KEY (player_one) REFERENCES users (player_hash),
	FOREIGN KEY (player_two) REFERENCES users (player_hash)
    )'''
    execute(cursor, conn, create_table_games)

def add_game(cursor, conn, player_one_auth, player_two_auth):
    query = '''INSERT INTO games (player_one, player_two, status) VALUES (
    (SELECT player_hash FROM users WHERE users.auth_token = '{p1auth}'),
    (SELECT player_hash FROM users WHERE users.auth_token = '{p2auth}'),
    'ongoing')'''.format(p1auth = player_one_auth, p2auth = player_two_auth)
    execute(cursor, conn, query)

def get_last_game(cursor):
    query = '''SELECT id FROM games ORDER BY id DESC LIMIT 1'''
    cursor.execute(query)
    return cursor.fetchall()

def get_opp_player_hash(cursor, game_id, player_no):
    if player_no == 1:
        query = '''SELECT player_one FROM games WHERE games.id = {}'''.format(game_id)
        key = 'player_one'
    else:
        query = '''SELECT player_two FROM games WHERE games.id = {}'''.format(game_id)
        key = 'player_two'
    cursor.execute(query)
    return cursor.fetchall()[0][key]

def update_game_status(cursor, conn, game_id, status):
    query = '''UPDATE games SET status = '{}' WHERE games.id = {} '''.format(status, game_id)
    execute(cursor, conn, query)

def end_game(cursor, conn, game_id):
    query = '''UPDATE games SET status = 'finished', ended = now() WHERE games.id = {game_id}'''.format(game_id = game_id)
    execute(cursor, conn, query)
