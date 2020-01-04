"""this is the test for zima-net"""


def test_build_network():
    import tensorflow as tf
    from net import network as zima_net
    print(zima_net((
        tf.placeholder(tf.uint8, [None, 8, 8, 4]),
        tf.placeholder(tf.int32, [None, ]),
        tf.placeholder(tf.int32, [None, ]),
        tf.placeholder(tf.float32, [None, ])
    )))


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
    from trainer import train_network_supervised
    from glob import glob
    train_network_supervised(glob('games_data/*.csv'),
                             'pokemon', num_epochs=60)


def test_data_generator():
    import tensorflow as tf
    from data_loader import get_batch
    from glob import glob
    batches, num_batches, tot_samples = get_batch(
        filenames=glob('games_data/*.csv'), batch_size=3)
    print(f'>>>>>>>>>>>>>>> batches: {batches}, num_batches: {num_batches}')

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
