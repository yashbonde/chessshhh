from chess_engine.chess_db.utils import execute

def create_moves_table(cursor, conn):
    create_table_games = '''CREATE TABLE moves (
	id SERIAL PRIMARY KEY NOT NULL,
	created time DEFAULT CURRENT_TIMESTAMP,
	deleted time,
	game_id int NOT NULL,
	player_hash char(32),
	game_state varchar (100),
	san varchar (10),
	FOREIGN KEY (game_id) REFERENCES games (id),
	FOREIGN KEY (player_hash) REFERENCES users (player_hash)
    )'''
    execute(cursor, conn, create_table_games)

def add_move_using_auth(cursor, conn, game_id, auth_token, board_config, san):
    query = '''INSERT INTO moves (game_id, player_hash, game_state, san) VALUES (
    {game_id}, (SELECT player_hash FROM users WHERE users.auth_token = '{auth_token}'),
    '{board_state}', '{san}'
    )'''.format(
        game_id = game_id,
        auth_token = auth_token,
        board_state = board_config,
        san = san
    )
    execute(cursor, conn, query)

def add_opp_move_using_auth(cursor, conn, player_no, game_id, auth_token, board_config, san):
    player_no_curr = 'player_one'
    player_no_opp = 'player_two'
    if player_no == 1:
        player_no_curr = 'player_two'
        player_no_opp = 'player_one'
    query = '''INSERT INTO moves (game_id, player_hash, game_state, san) VALUES (
        {game_id}, (SELECT games.{player_opp} FROM games WHERE games.{player_curr} =
        	(SELECT users.player_hash FROM users WHERE users.auth_token = '{auth_token}')
        	AND games.id = {game_id}),
        '{board_state}', '{san}')'''.format(
        game_id = game_id,
        auth_token = auth_token,
        player_curr = player_no_curr,
        player_opp = player_no_opp,
        board_state = board_config,
        san = san
    )
    execute(cursor, conn, query)
