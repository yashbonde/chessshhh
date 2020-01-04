import tensorflow as tf

def shapes_list(inp):
    """
    cleaner handling of tensorflow shapes
    :param inp: input tensor
    :return: list of shapes combining dynamic and static shapes
    """
    shapes_static = inp.get_shape().as_list()
    shapes_dynamic = tf.shape(inp)
    cleaned_shape = [shapes_dynamic[i] if s is None else s for i, s in enumerate(shapes_static)]
    return cleaned_shape

def ff(inp, num_features, scope, weights_init_stddev=0.2):
    """
    1D convolutional block, first reshape input then matmul weights and then reshape

    :param inp: input tensor
    :param scope: tf variable scope
    :param num_features: number of output features
    :param weights_init_stddev: standard deviation value
    :return: processed output
    """
    with tf.variable_scope(scope):
        *start, nx = shapes_list(inp)
        weights = tf.get_variable('w', [1, nx, num_features],
                                  initializer=tf.random_normal_initializer(stddev=weights_init_stddev))
        bias = tf.get_variable('b', [num_features],
                               initializer=tf.constant_initializer(0))

        # reshape input and weights and perform matmul and add bias
        inp_reshaped = tf.reshape(inp, [-1, nx])
        w_reshaped = tf.reshape(weights, [-1, num_features])
        out = tf.matmul(inp_reshaped, w_reshaped) + bias

        out = tf.reshape(out, start + [num_features])
        return out

def normalise_tensor(inp, scope, *, axis=-1, epsilon=1e-5):
    """
    Normalize the input values between 0 and 1, then do diagonal affine transform
    :param inp: input tensor
    :param scope: tf variable scope
    :param axis: axis to perform ops on
    :param epsilon: base minimum value
    :return: normalised tensor
    """
    with tf.variable_scope(scope):
        e_dim = inp.get_shape().as_list()[-1]
        g = tf.get_variable('g', [e_dim], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [e_dim], initializer=tf.constant_initializer(0))

        u = tf.reduce_mean(inp, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(inp - u), axis=axis, keepdims=True)
        out = (inp - u) * tf.rsqrt(s + epsilon)
        out = out * g + b

        return out