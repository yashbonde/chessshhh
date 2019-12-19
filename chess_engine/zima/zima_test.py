"""this is the test for zima-net"""

def test_build_network():
    from .net import network as zima_net
    print(zima_net())

def test_preprocess_states():
    from engine import make_state, preprocess_states
    boards = [
        '2r2rk1/4bp1p/bqn1p1pP/p3P3/1ppP1BN1/2p2NP1/P1P2P2/R1Q1R1K1 w - - 0 23',
        '2r2rk1/4bp1p/bqn1p1pP/p3P1B1/1ppP2N1/2p2NP1/P1P2P2/R1Q1R1K1 b - - 1 23',
        '2r2rk1/4bp1p/bq2p1pP/p3P1B1/1ppn2N1/2p2NP1/P1P2P2/R1Q1R1K1 w - - 0 24',
        '2r2rk1/4bp1p/bq2p1pP/p3P1B1/1ppN2N1/2p3P1/P1P2P2/R1Q1R1K1 b - - 0 24',
        '2r2rk1/4bp1p/b3p1pP/p3P1B1/1ppq2N1/2p3P1/P1P2P2/R1Q1R1K1 w - - 0 25'
    ]
    moves = [
        {'from': 29, 'to': 38, 'san': 'Bg5'},
        {'from': 42, 'to': 27, 'san': 'Nxd4'},
        {'from': 21, 'to': 27, 'san': 'Nxd4'},
        {'from': 41, 'to': 27, 'san': 'Qxd4'},
        {'from': 38, 'to': 52, 'san': 'Bxe7'}
    ]
    results = [
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0
    ]

    bstates, bfrom, bto, bvalues = preprocess_states(boards, moves, results)
    print('-------- STATES ----------\n{}'.format(bstates.shape))
    print('-------- MOVES (FROM) --------\n{}'.format(bfrom))
    print('-------- MOVES (TO) --------\n{}'.format(bto))
    print('-------- VALUES ---------\n{}'.format(bvalues))


def test_trainer():
    from trainer import train_network_supervised, load_data
    train_network_supervised('/Users/yashbonde/Desktop/AI/chessshhh/KB_small.pgn', 'pokemon', num_epochs = 100)

def test_data_generator():
    import tensorflow as tf
    from trainer import get_batch
    batches, num_batches = get_batch(fpath = '/Users/yashbonde/Desktop/AI/chessshhh/KB_small.pgn',
        batch_size = 3, shuffle = True
    )
    print(f'>>>>>>>>>>>>>>> batches: {batches}, num_batches: {num_batches}')

    # batches = get_batch(fpath = '/Users/yashbonde/Desktop/AI/chessshhh/KingBase2019-02.pgn',
    #     batch_size = 3, shuffle = True
    # )
    # print(f'batches: {batches}')

    # create a iterator of the correct shape and type
    iter = tf.data.Iterator.from_structure(
        batches.output_types, batches.output_shapes)
    xs = iter.get_next()
    train_init_op = iter.make_initializer(batches)  
    print(f'=============== xs: {xs}')  

    with tf.Session() as sess:
        sess.run(train_init_op)
        ops = (xs[0], xs[1], xs[2], xs[3])
        _board, _from, _to, _res = sess.run(ops)
        print(f'shape of board: {_board.shape}')
        print(f'from action: {_from}')
        print(f'to action: {_to}')
        print(f'result: {_res}')

if __name__ == "__main__":
    test_trainer()

