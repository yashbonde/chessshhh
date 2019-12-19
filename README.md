# Chessshhh App

This is a simple hacked app built to play chess. The player can chosse to play against a variety of difficult and scoring engines. Engines are trained in two phases, first is the blunt supervised learning kind of approach where it simply mimicks from the trained data and then through the concept of self play.

This is a simple educational app to teach me how to build a complete full stack deployable webapp. The steps for operation anf flow will be as follows:
1. User logs in and is then authenticated
2. Chessboard opens up and he can immidiately starts playing

Following are tables in the database we make:
1. `users`: the authentication table which also stores additional meta about the player
2. `games`: this database is used to store all the games palyed by different players
3. `moves: this is the flattened data about all the moves that are taken across different games
4. `robodata`: this has the information about different agents trained using self-play

# Usage

To run the app run the following on your command line:
```shell
source config.sh
python3 app.py
```

then goto `locahost:5000` on your browser, authenticate yourself and you are good to go!

# TODO
