"""this is the complete manager for all things DB
ideally this should be built and maintained using an ORM but
I tried that and it is becoming to much of a deviation and
pushing me away from actual delivery. So move to psycopg as a
quick hack.
"""

import os
import logging
import psycopg2
from psycopg2.extras import RealDictCursor

# custom
# from chess_engine.chess_db.games import create_games_table
# from chess_engine.chess_db.moves import create_moves_table
# from chess_engine.chess_db.users import create_users_table

def connect():
    connection = psycopg2.connect(user = os.environ["POSTGRES_USER"],
                                  password = os.environ["POSTGRES_PASSWORD"],
                                  host = os.environ["POSTGRES_HOST"],
                                  port = os.environ["POSTGRES_PORT"],
                                  database = os.environ["POSTGRES_DB"])
    cursor = connection.cursor(cursor_factory = RealDictCursor)
    return cursor, connection


def close_connection(cursor):
    cursor.close()


def execute(cursor, conn, command, log_ = True):
    print('>> Executing Query: {}'.format(command))
    cursor.execute(command)
    conn.commit()
