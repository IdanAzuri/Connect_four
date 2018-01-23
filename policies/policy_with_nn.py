import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


np.random.seed(1231)
from policies import base_policy as bp


NUM_ITERATIONS = 1000
GAMMA_FACTOR = 0.99
NUM_ACTIONS = 7

STATE_DIM = 7 * 6  # board size

INPUT_SIZE = STATE_DIM
FC1 = 256
FC2 = 128
FC3 = 64
FC4 = 32
EMPTY_VAL = 0
PLAYER1_ID = 1
PLAYER2_ID = 2
MAX_HISTORY_SIZE = 100
ROWS = 6
COLS = 7
WIN_MASK = np.ones(4)
ACTIONS = [0, 1, 2, 3, 4, 5, 6]


# parameters


# Consts


# Helper functions
def weight_variable(shape, name):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


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


class ExperienceReplay(object):
    """
    During gameplay all the experiences < s, a, r, s’ > are stored in a replay memory. 
    In training, batches of randomly drawn experiences are used to generate the input and target for training.
    """

    def __init__(self, session, x, y, y_logic, max_memory=100, discount=GAMMA_FACTOR):
        """
        Setup
        max_memory: the maximum number of experiences we want to store
        memory: a list of experiences
        discount: the discount factor for future experience

        In the memory the information whether the game ended at the state is stored seperately in a nested array
        [...
        [experience, game_over]
        [experience, game_over]
        ...]
        """
        self.x_input_ = x
        self.y_ = y
        self.session_ = session
        self.y_logic_ = y_logic
        self.max_memory = max_memory
        self.memory = list()  # TODO make dictionary
        self.discount = GAMMA_FACTOR

    def remember(self, states, game_over):
        # Save a state to memory, game over = 1 otherwise 0
        self.memory.append([states, game_over])
        # We don't want to store infinite memories, so if we have too many, we just delete the oldest one
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, batch_size=10):

        # How many experiences do we have?
        len_memory = len(self.memory)

        # Calculate the number of actions that can possibly be taken in the game
        num_actions = NUM_ACTIONS

        # Dimensions of the game field
        env_dim = INPUT_SIZE

        # We want to return an input and target vector with inputs from an observed state...
        inputs = np.zeros((min(len_memory, batch_size), env_dim))

        # ...and the target r + gamma * max Q(s’,a’)
        # Note that our target is a matrix, with possible fields not only for the action taken but also
        # for the other possible actions. The actions not take the same value as the prediction to not affect them
        targets = np.zeros((inputs.shape[0], num_actions))

        # We draw states to learn from randomly
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            """
            Here we load one transition <s, a, r, s’> from memory
            state_t: prev state s
            action_t: action taken a
            reward_t: reward earned r
            state_tp1: the state that followed s’
            """
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]

            # We also need to know whether the game ended at this state
            game_over = self.memory[idx][1]

            # add the state s to the input
            inputs[i:i + 1] = state_t


            # First we fill the target values with the predictions of the model.
            # They will not be affected by training (since the training loss for them is 0)
            # prev_action_predicted, predicted_actions_prob_vec
            prev_action_predicted, targets[i] = self.session_.run(self.y_logic_, self.y_, feed_dic={
                self.x_input_: state_t.reshape(
                    -1, STATE_DIM), self.y_: np.ones((1, 7))})[0]

            """
            If the game ended, the expected reward Q(s,a) should be the final reward r.
            Otherwise the target value is r + gamma * max Q(s’,a’)
            """
            #  Here Q_sa is max_a'Q(s', a')
            Q_sa = np.max(self.session_.predict(state_tp1)[0])

            # if the game ended, the reward is the final reward
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # r + gamma * max Q(s’,a’)
                targets[i, action_t] = reward_t + GAMMA_FACTOR * Q_sa
        return inputs, targets


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
            ("r", np.float32) #,
            # ("done", np.bool)
        ])
        self.clear()

    def clear(self):
        """Remove all entries from the DB."""

        self.index = 0
        self.n_items = 0
        self.full = False

    def store(self, s1, s2, a, r): #, done
        """Store new samples in the DB."""

        n = s1.shape[0]
        if self.index + n > self.db_size:
            self.full = True
            l = self.db_size - self.index
            if l > 0:
                self.store(s1[:l], s2[:l], a[:l], r[:l])  #, done[:l]
            self.index = 0
            if l < n:
                self.store(s1[l:], s2[l:], a[l:], r[l:])  #, done[l:]
        else:
            v = self.DB[self.index: self.index + n]
            v.s1 = s1
            v.s2 = s2
            v.a = a
            v.r = r
            # v.done = done
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
            # done = np.zeros(r.states.shape[1], np.bool)
            # done[-1] = True
            for i in range(r.states.shape[0]):
                s2 = np.vstack([r.states[i, 1:], self._empty_state])
                self.store(r.states[i], s2, r.actions[i], r.rewards[i]) #, done


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
    W1 = weight_variable([INPUT_SIZE, FC1], name="W1")
    B1 = bias_variable([FC1])
    W2 = weight_variable([FC1, FC2], name="W2")
    B2 = bias_variable([FC2])
    W3 = weight_variable([FC2, FC3], name="W3")
    B3 = bias_variable([FC3])
    W4 = weight_variable([FC3, FC4], name="W4")
    B4 = bias_variable([FC4])
    W_LAST_LAYER = weight_variable([FC4, NUM_ACTIONS], name="W_LAST_LAYER")
    B_LAST_LAYER = bias_variable([NUM_ACTIONS])

    # The model
    with tf.name_scope('reshape'):
        X_input = tf.reshape(X_input, [-1, INPUT_SIZE])
        Y1 = tf.nn.leaky_relu(tf.matmul(X_input, W1) + B1)
        Y2 = tf.nn.leaky_relu(tf.matmul(Y1, W2) + B2)
        Y3 = tf.nn.leaky_relu(tf.matmul(Y2, W3) + B3)
        Y4 = tf.nn.tanh(tf.matmul(Y3, W4) + B4)
        Y_logitis = tf.matmul(Y4, W_LAST_LAYER) + B_LAST_LAYER
        predict = tf.argmax(Y_logitis, 1)

    return Y_logitis, predict


class QLearningNetwork(bp.Policy):

    def cast_string_args(self, policy_args):
        # Example
        policy_args['depth'] = int(policy_args['depth']) if 'depth' in policy_args else 1
        return policy_args

    def init_run(self, save_path=None, l_rate=1e-2, session=None):

        self.log("Creating model...")
        self.g = tf.Graph()
        self.ex_replay = ExperienceReplay(MAX_HISTORY_SIZE)
        with self.g.as_default():
            self.session = tf.Session() if session is None else session
            self.saver = None
            # self.saver.restore(self.session, "/tmp/model.ckpt")
            # self.log("The model restored successfully! W3={}".format(tf.get_variable("W3")))

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
            self.load(path=None)
            # self.session.run(tf.global_variables_initializer())

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        # find legal actions:

        legal_actions = np.array(np.where(new_state[0, :] == EMPTY_VAL))
        legal_actions = np.reshape(legal_actions, (legal_actions.size,))
        legal_actions_hot_vector = np.zeros((7,), dtype=np.int32)
        legal_actions_hot_vector[legal_actions] = 1
        if len(legal_actions) == 0:  # maybe to learn a draw?
            return

        state_after_predicted_action = new_state
        self.log("Actions={}".format(legal_actions_hot_vector), "DEBUG")
        # [not sure necessary] ]in case of weird problems and draws (no legal actions):
        prev_action_predicted = 0
        reward_for_predicted_action = -1  # punishment for illegal cation
        predicted_actions_prob_vec = np.random.randn(1, 7)
        all_rewards = 0
        is_win = 0
        # The Q-Network
        # Choose an action by greedily (with e chance of random action) from the Q-network
        try:
            prev_action_predicted, predicted_actions_prob_vec = \
                self.session.run([self.y_argmax, self.y_logitis],
                                 feed_dict={
                                     self.x_input: prev_state.reshape(
                                         -1, STATE_DIM),
                                     self.y: np.ones((1, 7))})
            if np.random.rand(1) < self.epsilon:  # exploration
                prev_action_predicted = np.random.choice(legal_actions)  # random action
            # Get new state and reward from environment
            elif int(prev_action_predicted) not in legal_actions:
                prev_action_predicted = np.random.choice(legal_actions)  # random action
            self.log("Real action:{}, predicted action:{}".format(prev_action, prev_action_predicted), "DEBUG")
            if int(prev_action_predicted) in legal_actions:  # if the action is ilegal learn the real game action and
                #  reward
                state_after_predicted_action = make_move(prev_state, prev_action_predicted,
                                                         self.id)  # get new state for the action

                is_win = check_for_win(state_after_predicted_action, self.id, int(prev_action_predicted))
                reward_for_predicted_action = int(is_win)
                if reward_for_predicted_action < reward:  # penalized if you could win but you didn't win
                    reward_for_predicted_action = -1

            # Obtain the Q' values by feeding the new state through our network
            actions_prob_vec_after_playing = self.session.run(self.y_logitis,
                                                              feed_dict={
                                                                  self.x_input: state_after_predicted_action.reshape(-1,
                                                                                                                     INPUT_SIZE),
                                                                  self.y: predicted_actions_prob_vec})
            # Obtain maxQ' and set our target value for chosen action.
            max_action_prob_after_playing = np.max(actions_prob_vec_after_playing)
            self.log("prob vector={}".format(actions_prob_vec_after_playing), "DEBUG")
            self.log("predicted_actions_prob_vec before boost prob vector={}".format(predicted_actions_prob_vec),
                     "DEBUG")
            predicted_actions_prob_vec[0, prev_action_predicted] = \
                reward_for_predicted_action + GAMMA_FACTOR * max_action_prob_after_playing
            if reward == -1:
                predicted_actions_prob_vec[0, prev_action] = -1 + GAMMA_FACTOR * predicted_actions_prob_vec[
                    0, prev_action]
                self.log("PUNISHED for action={}".format(prev_action))
            self.log("Boosted prob vector={}".format(predicted_actions_prob_vec), "DEBUG")
            self.log("is_win={},reward={},reward_for_predicted_action={}\n".format(is_win, reward,
                                                                                   reward_for_predicted_action),
                     "DEBUG")
            # Trying Temporal_difference_learning
            # https://en.wikipedia.org/wiki/Temporal_difference_learning
            # predicted_actions_prob_vec[
            #     0, prev_action_predicted] = GAMMA_FACTOR * (
            #         reward_for_predicted_action + max_action_prob_after_playing) - np.max(predicted_actions_prob_vec)
            # Train our network using target and predicted Q values
            self.session.run([self.trainer, self.loss],
                             feed_dict={self.x_input: new_state.reshape(-1, INPUT_SIZE),
                                        self.y: predicted_actions_prob_vec})
            # Learnig from real game parameters TODO combine it to a dictionary to feed the model once
            # Learning wins
            # if reward == 1:
            #     real_prob_vector =np.zeros((1,7))
            #     real_prob_vector[prev_action] = 1  # set the win in 1 hot vector
            #     self.session.run([self.trainer, self.loss],
            #                      feed_dict={self.x_input: new_state.reshape(-1, INPUT_SIZE),
            #                                 self.y: real_prob_vector})
            all_rewards += reward_for_predicted_action
            new_state = state_after_predicted_action
            # TODO ADD LOOP TO LEARN THE NEW STATE WHILE THE GAME ISN'T OVER

            # self.steps_list.append(j)
            self.rewards_list.append(all_rewards)
            print(" SCORE: " + str(sum(self.rewards_list)))
            if self.rewards_list == 10:
                self.save_model()
        except ValueError as e:
            self.log("ValueError error({0})".format(e), "ERROR")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
            print("prev_state:{}, prev_action_predicted:{}".format(prev_state, prev_action_predicted))
        except IOError as e:
            self.log("I/O error({0}): {1}".format(e.errno, e.strerror), "ERROR")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
        except IndexError as e:
            self.log("IndexError,prev_state:{}, prev_action_predicted:{}, error({0})".format(prev_state,
                                                                                             prev_action_predicted,
                                                                                             e), "ERROR")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
        except:
            self.log("Unexpected error:{}".format(sys.exc_info()[0]), "ERROR")
            raise

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        temp_actions = np.random.randn(1, 7)
        legal_actions = np.array(np.where(new_state[0, :] == EMPTY_VAL))
        legal_actions = np.reshape(legal_actions, (legal_actions.size,))
        # legal_actions_hot_vector = np.zeros((7,), dtype=np.int32)
        # legal_actions_hot_vector[legal_actions] = 1
        # TODO save  to replayDB:{round, prev_state, prev_action, reward, new_state}
        # Here we load one transition <s, a, r, s’> from memory
        self.ex_replay.remember([prev_state, prev_action, reward, new_state], bool(reward))
        action = \
            self.session.run(self.y_argmax, feed_dict={self.x_input: new_state.reshape(-1, STATE_DIM),
                                                       self.y: temp_actions})[0]

        if action in legal_actions:  # and np.random.random() > self.epsilon:
            return action
        else:
            return np.random.choice(legal_actions)

    def load(self, path="/tmp/model_connect_4/"):
        """Load weights or init variables if path==None."""

        if self.saver is None:
            self.saver = tf.train.Saver(max_to_keep=None)

        if path is None or (os.path.exists(path) and not os.listdir(path)):
            self.session.run(tf.global_variables_initializer())
            return 0
        elif path is not None:
            p = Path(path)

            files = p.glob("**/model.ckpt.meta")
            # newest = max(files, key=lambda p: p.stat().st_ctime)
            # fname = str(newest)[:-5]
            fname = "./tmp/model.ckpt.meta"
            self.saver.restore(self.session, fname)

            return 1  # int(newest.parts[-2])

    def save_model(self, save_path="/tmp/model_connect_4/"):
        """Save the current graph."""

        if self.saver is None:
            with self.g.as_default():
                self.saver = tf.train.Saver(max_to_keep=None)

        p = Path(save_path)
        p.mkdir(parents=True, exist_ok=True)

        fname = str(p / "{:04d}".format(sum(self.rewards_list)) / "model.ckpt")
        self.saver.save(self.session, fname)
        self.log("Model saved in file: %s" % save_path)

        return
