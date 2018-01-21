from pathlib import Path

import numpy as np
import tensorflow as tf

from policies import base_policy as bp


NUM_ITERATIONS = 1000
FACTOR = 0.99
NUM_ACTIONS = 7

STATE_DIM = 7 * 6  # board size

INPUT_SIZE = STATE_DIM
FC1 = 30
FC2 = 20
FC3 = 60
EMPTY_VAL = 0
PLAYER1_ID = 1
PLAYER2_ID = 2

ROWS = 6
COLS = 7
WIN_MASK = np.ones(4)
ACTIONS = [0, 1, 2, 3, 4, 5, 6]

# parameters
epsilon = .1  # exploration
num_actions = 3  # [move_left, stay, move_right]
max_memory = 500  # Maximum number of experiences we are storing
hidden_size = 100  # Size of the hidden layers
batch_size = 1  # Number of experiences we use for training per batch
grid_size = 10  # Size of the playing field


# Consts


# Helper functions
def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class NeuralNetwork:

    @staticmethod
    def run_op_in_batches(session, op, batch_dict={}, batch_size=None,
                          extra_dict={}):

        """Return the result of op by running the network on small batches of
        batch_dict."""

        if batch_size is None:
            return session.run(op, feed_dict={**batch_dict, **extra_dict})

        # Probably the least readable form to get an arbitrary item from a dict
        n = len(next(iter(batch_dict.values())))

        s = []
        for i in range(0, n, batch_size):
            bd = {k: b[i: i + batch_size] for (k, b) in batch_dict.items()}
            s.append(session.run(op, feed_dict={**bd, **extra_dict}))

        if s[0] is not None:
            if np.ndim(s[0]):
                return np.concatenate(s)
            else:
                return np.asarray(s)

    def __init__(self, input_dim, output_dim, hidden_layers, session=None,
                 name_prefix="", input_=None):
        """Create an ANN with fully connected hidden layers of width
        hidden_layers."""

        self.saver = None

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name_prefix = name_prefix

        self.weights = []
        self.biases = []

        self.session = tf.Session() if session is None else session
        if input_ is None:
            self.input = tf.placeholder(tf.float32,
                                        shape=(None, self.input_dim),
                                        name="{}input".format(self.name_prefix)
                                        )
        else:
            self.input = input_
        self.layers = [self.input]

        for i, width in enumerate(hidden_layers):
            a = self.affine("{}hidden{}".format(self.name_prefix, i),
                            self.layers[-1], width)
            self.layers.append(a)

        self.output = self.affine("{}output".format(self.name_prefix),
                                  self.layers[-1], self.output_dim, relu=False)
        self.probabilities = tf.nn.softmax(self.output,
                                           name="{}probabilities".format(self.name_prefix))
        self.output_max = tf.reduce_max(self.output, axis=1)
        self.output_argmax = tf.argmax(self.output, axis=1)

    def vars(self):
        """Iterate over all the variables of the network."""

        for w in self.weights:
            yield w
        for b in self.biases:
            yield b

    def affine(self, name_scope, input_tensor, out_channels, relu=True,
               residual=False):
        """Create a fully-connected affaine layer."""

        input_shape = input_tensor.get_shape().as_list()
        input_channels = input_shape[-1]
        with tf.variable_scope(name_scope):
            W = tf.get_variable("weights",
                                initializer=tf.truncated_normal(
                                    [input_channels, out_channels],
                                    stddev=1.0 / np.sqrt(float(input_channels))
                                ))
            b = tf.get_variable("biases",
                                initializer=tf.zeros([out_channels]))

            self.weights.append(W)
            self.biases.append(b)

            A = tf.matmul(input_tensor, W) + b

            if relu:
                R = tf.nn.relu(A)
                if residual:
                    return R + input_tensor
                else:
                    return R
            else:
                return A

    def take(self, indices):
        """Return an operation that takes values from network outputs.
        e.g. NN.predict_max() == NN.take(NN.predict_argmax())
        """

        mask = tf.one_hot(indices=indices, depth=self.output_dim, dtype=tf.bool,
                          on_value=True, off_value=False, axis=-1)
        return tf.boolean_mask(self.output, mask)

    def assign(self, other):
        """Return a list of operations that copies other network into self."""

        ops = []
        for (vh, v) in zip(self.vars(), other.vars()):
            ops.append(tf.assign(vh, v))
        return ops

    def reinit(self):
        """Reset weights to initial random values."""

        for w in self.weights:
            self.session.run(w.initializer)
        for b in self.biases:
            self.session.run(b.initializer)

    def save(self, save_path, step):
        """Save the current graph."""

        if self.saver is None:
            with self.g.as_default():
                self.saver = tf.train.Saver(max_to_keep=None)

        p = Path(save_path)
        p.mkdir(parents=True, exist_ok=True)

        fname = str(p / "{:04d}".format(step) / "model.ckpt")
        self.saver.save(self.session, fname)

        return p / "{:04d}".format(step)

    def load(self, path):
        """Load weights or init variables if path==None."""

        if self.saver is None:
            self.saver = tf.train.Saver(max_to_keep=None)

        if path is None:
            self.session.run(tf.global_variables_initializer())
            return 0
        else:
            p = Path(path)

            files = p.glob("**/model.ckpt.meta")
            newest = max(files, key=lambda p: p.stat().st_ctime)
            fname = str(newest)[:-5]

            self.saver.restore(self.session, fname)

            return int(newest.parts[-2])

    def predict_probabilities(self, inputs_feed, batch_size=None):
        """Return softmax on NN outputs."""

        feed_dict = {self.input: inputs_feed}
        return self.run_op_in_batches(self.session, self.probabilities,
                                      feed_dict, batch_size)

    def predict_argmax(self, inputs_feed, batch_size=None):
        """Return argmax on NN outputs."""

        feed_dict = {self.input: inputs_feed}
        return self.run_op_in_batches(self.session, self.output_argmax,
                                      feed_dict, batch_size)

    def predict_max(self, inputs_feed, batch_size=None):
        """Return max on NN outputs."""

        feed_dict = {self.input: inputs_feed}
        return self.run_op_in_batches(self.session, self.output_max,
                                      feed_dict, batch_size)

    def predict_raw(self, inputs_feed, batch_size=None):
        """Return NN outputs without transformation."""

        feed_dict = {self.input: inputs_feed}
        return self.run_op_in_batches(self.session, self.output,
                                      feed_dict, batch_size)

    def predict_random(self, inputs_feed, epsilon=0.01, batch_size=None):
        """Return random element based on softmax on the NN outputs.
        epsilon is a smoothing parameter."""

        n = len(inputs_feed)
        base = self.predict_probabilities(inputs_feed, batch_size) + epsilon
        probs = base / base.sum(1, keepdims=True)
        out = np.zeros(n, np.int32)

        for i in range(n):
            out[i] = np.random.choice(self.output_dim, 1, p=probs[i])

        return out

    def predict_exploration(self, inputs_feed, epsilon=0.1, batch_size=None):
        """Return argmax with probability (1-epsilon), and random value with
        probabilty epsilon."""

        n = len(inputs_feed)
        out = self.predict_argmax(inputs_feed, batch_size)
        exploration = np.random.random(n) < epsilon
        out[exploration] = np.random.choice(self.output_dim, exploration.sum())

        return out

    def train_in_batches(self, train_op, feed_dict, n_batches, batch_size,
                         balanced=False):
        """Train the network by randomly sub-sampling feed_dict."""

        keys = tuple(feed_dict.keys())
        if balanced:
            ds = BalancedDataSet(*[feed_dict[k] for k in keys])
        else:
            ds = DataSet(*[feed_dict[k] for k in keys])

        for i in range(n_batches):
            batch = ds.next_batch(batch_size)
            d = {k: b for (k, b) in zip(keys, batch)}
            self.session.run(train_op, d)

    def accuracy(self, accuracy_op, feed_dict, batch_size):
        """Return the average value of an accuracy op by running the network
        on small batches of feed_dict."""

        return self.run_op_in_batches(self.session, accuracy_op,

                                      feed_dict, batch_size).mean()


class ReplayDB:
    """Holds previous games and allows sampling random combinations of
        (state, action, new state, reward)
    """

    def __init__(self, state_dim, db_size):
        """Create new DB of size db_size."""

        self.state_dim = state_dim
        self.db_size = db_size
        self._empty_state = np.zeros((1, self.state_dim))

        self.DB = np.rec.recarray(self.db_size, dtype=[
            ("s1", np.float32, self.state_dim),
            ("s2", np.float32, self.state_dim),
            ("a", np.int32),
            ("r", np.float32),
            ("done", np.bool)
        ])
        self.clear()

    def clear(self):
        """Remove all entries from the DB."""

        self.index = 0
        self.n_items = 0
        self.full = False

    def store(self, s1, s2, a, r, done):
        """Store new samples in the DB."""

        n = s1.shape[0]
        if self.index + n > self.db_size:
            self.full = True
            l = self.db_size - self.index
            if l > 0:
                self.store(s1[:l], s2[:l], a[:l], r[:l], done[:l])
            self.index = 0
            if l < n:
                self.store(s1[l:], s2[l:], a[l:], r[l:], done[l:])
        else:
            v = self.DB[self.index: self.index + n]
            v.s1 = s1
            v.s2 = s2
            v.a = a
            v.r = r
            v.done = done
            self.index += n

        self.n_items = min(self.n_items + n, self.db_size)

    def sample(self, sample_size=None):
        """Get a random sample from the DB."""

        if self.full:
            db = self.DB
        else:
            db = self.DB[:self.index]

        if (sample_size is None) or (sample_size > self.n_items):
            return db
        else:
            return np.rec.array(np.random.choice(db, sample_size, False))

    def iter_samples(self, sample_size, n_samples):
        """Iterate over random samples from the DB."""

        if sample_size == 0:
            sample_size = self.n_items

        ind = self.n_items
        for i in range(n_samples):
            end = ind + sample_size
            if end > self.n_items:
                ind = 0
                end = sample_size
                p = np.random.permutation(self.n_items)
                db = np.rec.array(self.DB[p])
            yield db[ind: end]
            ind = end

    def store_episodes_results(self, results):
        """Store all results from episodes (in the format of
        greenlet_learner.)"""

        for r in results:
            done = np.zeros(r.states.shape[1], np.bool)
            done[-1] = True
            for i in range(r.states.shape[0]):
                s2 = np.vstack([r.states[i, 1:], self._empty_state])
                self.store(r.states[i], s2, r.actions[i], r.rewards[i], done)


class DataSet:
    """A class for datasets (labeled data). Supports random batches."""

    def __init__(self, *args):
        """Create a new dataset."""

        self.X = [a.copy() for a in args]
        self.n = self.X[0].shape[0]
        self.ind = 0
        self.p = np.random.permutation(self.n)

    def next_batch(self, batch_size):
        """Get the next batch of size batch_size."""

        if batch_size > self.n:
            batch_size = self.n

        if self.ind + batch_size > self.n:
            # we reached end of epoch, so we shuffle the data
            self.p = np.random.permutation(self.n)
            self.ind = 0

        batch = self.p[self.ind: self.ind + batch_size]
        self.ind += batch_size

        return tuple(a[batch] for a in self.X)


class BalancedDataSet:
    """A class for datasets (labeled data). Supports balanced random batches."""

    def __init__(self, X, l):
        """Create a new dataset."""

        labels = set(l)
        self.n_groups = len(labels)
        self.groups = []

        for label in labels:
            X_i = X[l == label]
            ds_i = DataSet(X_i, np.repeat(label, X_i.shape[0]))
            self.groups.append(ds_i)

        self.n = min(ds.n for ds in self.groups) * self.n_groups

    def next_batch(self, batch_size):
        """Get the next batch of size batch_size."""

        group_size = batch_size // self.n_groups

        X = []
        l = []

        for group in self.groups:
            X_i, l_i = group.next_batch(group_size)
            X.append(X_i)
            l.append(l_i)

        return np.vstack(X), np.hstack(l)


def check_for_win(board, player_id, col):
    """
    check the board to see if last move was a winning move.
    :param board: the new board
    :param player_id: the player who made the move
    :param col: his action
    :return: True iff the player won with his move
    """

    row = 0

    # check which row was inserted last:
    for i in range(ROWS):
        if board[ROWS - 1 - i, col] == EMPTY_VAL:
            row = ROWS - i
            break

    # check horizontal:
    vec = board[row, :] == player_id
    if np.any(np.convolve(WIN_MASK, vec, mode="valid") == 4):
        return True

    # check vertical:
    vec = board[:, col] == player_id
    if np.any(np.convolve(WIN_MASK, vec, mode="valid") == 4):
        return True

    # check diagonals:
    vec = np.diagonal(board, col - row) == player_id
    if np.any(np.convolve(WIN_MASK, vec, mode="valid") == 4):
        return True
    vec = np.diagonal(np.fliplr(board), COLS - col - 1 - row) == player_id
    if np.any(np.convolve(WIN_MASK, vec, mode="valid") == 4):
        return True

    return False


def make_move(board, action, player_id):
    """
    return a new board with after performing the given move.
    :param board: original board
    :param action: move to make (column)
    :param player_id: player that made the move
    :return: new board after move was made
    """
    row = np.max(np.where(board[:, action] == EMPTY_VAL))
    new_board = np.copy(board)
    new_board[row, action] = player_id

    return new_board


def deep_nn(X_input):
    W1 = weight_variable([INPUT_SIZE, FC1])
    B1 = bias_variable([FC1])
    W2 = weight_variable([FC1, FC2])
    B2 = bias_variable([FC2])
    W3 = weight_variable([FC2, NUM_ACTIONS])
    B3 = bias_variable([NUM_ACTIONS])

    # The model
    with tf.name_scope('reshape'):
        X_input = tf.reshape(X_input, [-1, INPUT_SIZE])
        Y1 = tf.nn.relu(tf.matmul(X_input, W1) + B1)
        Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
        Y_logitis = tf.matmul(Y2, W3) + B3
        predict = tf.argmax(Y_logitis, 1)

    return Y_logitis, predict


class QLearningNetwork(bp.Policy):

    def cast_string_args(self, policy_args):

        policy_args['depth'] = int(policy_args['depth']) if 'depth' in policy_args else 1
        return policy_args

    def init_run(self, save_path=None, l_rate=1e-2):

        self.log("Creating model...")

        self.g = tf.Graph()
        with self.g.as_default():
            # tf Graph Input
            self.x_input = tf.placeholder(tf.float32, [None, STATE_DIM], name='input_data')
            # y is the next Q
            self.y = tf.placeholder(tf.float32, [None, NUM_ACTIONS], name='predicted')
            # predicted Q
            self.y_logitis, self.y_argmax = deep_nn(self.x_input)

            self.loss = tf.reduce_sum(tf.square(self.y - self.y_logitis))
            self.trainer = tf.train.GradientDescentOptimizer(learning_rate=l_rate).minimize(self.loss)

            # create lists to contain total rewards and steps per episode
            self.steps_list = []
            self.rewards_list = []
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())

        return

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        for i in range(NUM_ITERATIONS):
            # find legal actions:
            legal_actions = np.array(np.where(new_state[0, :] == EMPTY_VAL))
            legal_actions = np.reshape(legal_actions, (legal_actions.size,))

            # [not sure necessary] ]in case of weird problems and draws (no legal actions):
            if len(legal_actions) == 0:
                return 0, 0
            all_rewards = 0
            d = False
            j = 0
            # The Q-Network
            while j < 98:
                j += 1
                # Choose an action by greedily (with e chance of random action) from the Q-network
                a, allQ = self.session.run([self.y_argmax, self.y_logitis],
                                           feed_dict={self.x_input: np.identity(STATE_DIM)[new_state:new_state + 1]})
                if np.random.rand(1) < self.epsilon:
                    a = np.random.choice(legal_actions)  # random action
                # Get new state and reward from environment
                s1 = make_move(new_state, a, self.id)  # get new state for the action
                r = int(check_for_win(s1, self.id, a))  # reward
                # Obtain the Q' values by feeding the new state through our network
                Q1 = self.session.run(self.y_argmax, feed_dict={self.x_input: np.identity(16)[s1:s1 + 1]})
                # Obtain maxQ' and set our target value for chosen action.
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0, a[0]] = r + FACTOR * maxQ1
                # Train our network using target and predicted Q values
                self.session.run([self.trainer, self.loss],
                                 feed_dict={self.x_input: np.identity(STATE_DIM)[s:s + 1], self.y_logitis: targetQ})
                all_rewards += r
                s = s1
                if d == True:
                    # Reduce chance of random action as we train the model.
                    e = 1. / ((i / 50) + 10)
                    break
            self.steps_list.append(j)
            self.rewards_list.append(all_rewards)
        print("Percent of succesful episodes: " + str(sum(self.rewards_list) / NUM_ITERATIONS) + "%")

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        legal_actions = np.array(np.where(new_state[0, :] == EMPTY_VAL))
        legal_actions = np.reshape(legal_actions, (legal_actions.size,))

        action = self.session.run(self.y_argmax, feed_dict={self.x_input: legal_actions[None, :]})[0]

        if action in legal_actions and np.random.random() > self.epsilon:
            return action
        else:
            return np.random.choice(legal_actions)

    def save_model(self):

        return [self.session.run(self.W), self.session.run(self.b)], None