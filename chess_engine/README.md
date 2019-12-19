# Chess Engine

This is the rant I am doign as I code up the entire NN stack from scratch in next 10+ hours. There are a couple good soures that I am consulting from for building this, I was stuck on the network architecture for some time. Sources:
* [A blog](https://erikbern.com/2014/11/29/deep-learning-for-chess.html)
* [Chess Paper](https://pdfs.semanticscholar.org/28a9/fff7208256de548c273e96487d750137c31d.pdf)
* [Another blog](https://towardsdatascience.com/predicting-professional-players-chess-moves-with-deep-learning-9de6e305109e)
* [Self Play blog](https://medium.com/applied-data-science/how-to-build-your-own-alphazero-ai-using-python-and-keras-7f664945c188)
* [Code from Geohot](https://github.com/geohot/twitchchess)

## State

We first have to define what are state is and what is that we will feed to the neural network. So we have 7 pieces on the board, `Pawn (p|P), Rook (r|R), Knight (n|N), Bishop (b|B), King (k|K), Queen (q|Q) + noPiece`. Which can be represented on the board in 6 separate layer (total will be a `8x8x6` state space). In each of the layer we have a `+1` value where our unit is and a `-1` value where the opponents same unit is.

Other than these units we have the additional states such as King/Queenside castling rights + en passant. Each state can be represented in [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) as follows:
```
rnbqkb1r/ppp2ppp/3p4/8/8/2n2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 6
------------------------1----------------------- 2 -3-- 4 5 6

1: board state in a string
2: whose is the next turn
3: King/Queenside castling rights
4: en passant location
5: halfmove clock -> number of moves since last pawn capture
6: fullmove number -> increments when black plays
```

Thus there are following ways to keep the state information:
* `8x8x6`: Only consider the positions and no other additional information
* `8x8x7`: Considering the positions + en passant position + castling information on the four rook positions
* `8x8x4`: Considering the positiosn + castling rights (stored in binary, something like what @geohot did). However we will be performing data augmentation to make black play as white and we do not need an additional layer with just information about whose turn it is.

## Actions

Since the actions become dynamic in nature, unlike the ATARI games we cannot have a policy network that gives the probabilities to different actions. To deal with this we use a tree-search algorithm where each leaf is evaluated using our value network, which gives the probability of winning or losing the game.

The network thus gives us two separate probabilities one for `from_square` and other gives `to_square` probabilities. Once we get them we filter only the legal moves and then sort them by sum of probabilities. This then gives us how preferred each move will be. We can then start pruning them by selecting a `beam_size` which tells only go further in these `beam_size` moves and see the possibilities of each one of them.

## Network

Network will be a fairly simple ConvNet with two heads, one for `from_square` and other for `to_square`. It will be made in `tensorflow` as that is the best I know how to use. Fuck you!

## TODO

Things to do:

1. Convert the loader to a tf.data Thing
2. Start training first generation
3. Start looking at RL algorithms to train the network
4. Change the front end to start a new game if need be

```re
\[(Event\s\").*\n.*(\n\n)(1.).*((\n).*)*
\[(Event\s\".*)(\n.*)*[0|1|2]\n
```