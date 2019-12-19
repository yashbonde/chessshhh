"""
This is the ai_move.py file which has the AI model

17.11.2019 - @yashbonde
"""

# chess
import chess

# custom
from chess_engine.ai_model import make_random_move
from chess_engine.utils import board_position_to_int, int_to_board_position

def move_orchestrator(prev_board_state):
   """

   takes in the previous board state and returns object with keys
   {
      "new_state",
      "from",
      "to",
   }
   :param prev_board_state:
   :param move_obj:
   :return:
   """

   # default setup
   from_ = None
   to_ = None
   content = None
   san = None,
   new_state = prev_board_state

   # to understand the FEN: https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
   # feed the state to the model and get the move and updated state
   move_obj, game_state, san_move = make_random_move(
      board_state_fen = prev_board_state
   )

   # --- Write Outputs --- #
   if move_obj is 'checkmate':
      new_state = game_state
      content = "Checkmate, You Lost! Fucking Loser!"
   
   elif move_obj is "stalemate":
      new_state = game_state
      content = "Stalemate! Game Over Asshole!"

   else:
      new_state = game_state
      from_ = int_to_board_position[move_obj.from_square]
      to_ = int_to_board_position[move_obj.to_square]
      san = san_move
   
   # --- output body --- #
   return {
      "new_state": new_state,
      "from": from_,
      "to": to_,
      "content": content,
      "san": san
   }
