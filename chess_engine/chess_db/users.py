from chess_engine.chess_db.utils import execute

def create_users_table(cursor, conn):
    create_table_users = '''CREATE TABLE users (
        id SERIAL PRIMARY KEY NOT NULL,
        username varchar (50) UNIQUE NOT NULL,
        password varchar (50) NOT NULL,
        created time DEFAULT CURRENT_TIMESTAMP,
        num_games int,
        num_sessions int,
        player_hash char(32) UNIQUE NOT NULL,
        won_as_white int,
        won_as_black int,
        lost_as_white int,
        lost_as_black int,
        draw_as_white int,
        draw_as_black int,
        auth_token varchar (36) UNIQUE
        )'''
    execute(cursor, conn, create_table_users)


def add_user(cursor, conn, username, password, player_hash):
    query = '''INSERT INTO users (username, password, player_hash) VALUES ('{}', '{}', '{}')'''.format(username, password, player_hash)
    execute(cursor, conn, query)

# def update_game_result(cursor, conn, auth_token, what):
    # if what not in ['won_as_white', 'won_as_black', 'lost_as_white', 'lost_as_black', 'draw_as_white', 'draw_as_black']


def update_auth_token(cursor, conn, username, auth_token):
    query = '''UPDATE users SET auth_token = '{}' WHERE users.username = '{}' '''.format(auth_token, username)
    execute(cursor, conn, query)


def update_session(cursor, conn, auth_token):
    query = '''SELECT num_sessions FROM users WHERE users.auth_token = '{}' '''.format(auth_token)
    cursor.execute(query)
    num_session = cursor.fetchall()[0]['num_sessions']
    if not num_session:
        num_session = 1
    else:
        num_session += 1
    query = '''UPDATE users SET num_sessions = {} WHERE users.auth_token = '{}' '''.format(num_session, auth_token)
    execute(cursor, conn, query)


def get_users(cursor, username, password):
    query = '''SELECT player_hash FROM users WHERE users.username = '{}' AND users.password = '{}' '''.format(username, password)
    cursor.execute(query)
    return cursor.fetchall()


def get_player_hash_for_auth_token(cursor, auth_token):
    query = '''SELECT player_hash FROM users WHERE users.auth_token = '{}' '''.format(auth_token)
    cursor.execute(query)
    return cursor.fetchall()