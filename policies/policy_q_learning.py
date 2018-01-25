import os
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf

np.random.seed(1231)
from policies import base_policy as bp

LEANING_RATE = 1e-2
BATCH_SIZE = 4
GAMMA_FACTOR = 0.99
NUM_ACTIONS = 7
STATE_DIM = 7 * 6  # board size
INPUT_SIZE = STATE_DIM
FC1 = 512
FC2 = 128
FC3 = 64
# FC4 = 32
EMPTY_VAL = 0
PLAYER1_ID = 1
PLAYER2_ID = 2
ROWS = 6
COLS = 7
WIN_MASK = np.ones(4)


class ExperienceReplay(object):
    """
    During gameplay all the experiences < s, a, r, s’ > are stored in a replay memory.
    In training, batches of randomly drawn experiences are used to generate the input and target for training.
    """

    def __init__(self, max_memory=1e7, discount=GAMMA_FACTOR):
        self.max_memory = max_memory
        self.memory = []
        self.discount = discount
        self.memory_last_move = []  # Balancing the dataset
        self.max_memory_last_move = max_memory
        self.memory_len = 0

    def store(self, states):
        # Save a state to memory, game over = 1 otherwise 0
        self.memory.append([states])
        # We don't want to store infinite memories, so if we have too many, we just delete the oldest one
        if len(self.memory) > self.max_memory:
            del self.memory[:100]

    def inverse_last_move(self, states):
        s1, action, reward, s2 = states
        flip_s1 = s1 * 2 % 3
        flip_s2 = s2 * 2 % 3
        reward *= -1
        return [flip_s1, action, reward, flip_s2]

    def store_last_move(self, states):
        # Save a state to memory, game over = 1 otherwise 0
        # will learn also the inverse last move
        flip_states = self.inverse_last_move(states)
        self.memory_last_move.append([flip_states])
        self.memory_last_move.append([states])
        # We don't want to store infinite memories, so if we have too many, we just delete the oldest one
        if len(self.memory_last_move) > self.max_memory:
            del self.memory_last_move[:100]

    def get_batch(self, batch_size=32):
        """
        Here we load one transition <s, a, r, s’> from memory

        :param batch_size:

        :return: a permutation of:
        state_t: prev state s
        action_t: action taken a
        reward_t: reward earned r
        state_tp1: the state that followed s’
        return state_t, action_t, reward_t, state_tp1
        """
        self.memory_len = len(self.memory)
        shuffle_indices = np.random.permutation(min(self.memory_len, batch_size, int(batch_size)))
        return np.asarray(self.memory)[shuffle_indices]

    def get_last_move_batch(self, batch_size=32):
        """
        Here we load one transition <s, a, r, s’> from memory

        :param batch_size:

        :return: a permutation of:
        state_t: prev state s
        action_t: action taken a
        reward_t: reward earned r
        state_tp1: the state that followed s’
        return state_t, action_t, reward_t, state_tp1
        """
        last_move_len = len(self.memory_last_move)
        last_move_shuffle_indices = np.random.permutation(min(last_move_len, int(batch_size)))
        return np.asarray(self.memory_last_move)[last_move_shuffle_indices]

    def get_balanced_batch(self, batch_size=32):
        """
        Get balanced last move samples and regular samples
        :param batch_size:
        :return: 2 batches
        """
        batch_size /= 2  # divding batch size equally
        batch_samples = []
        last_move_len = len(self.memory_last_move)
        self.memory_len = len(self.memory)

        # sampling batch/2 last move states
        last_move_shuffle_indices = np.random.permutation(min(last_move_len, int(batch_size)))
        batch_samples.append(np.asarray(self.memory_last_move)[last_move_shuffle_indices])

        # sampling batch/2 regular game
        shuffle_indices = np.random.permutation(min(int(batch_size), self.memory_len))
        np.random.shuffle(shuffle_indices)  # get shuffled batch
        batch_samples.append(np.asarray(self.memory)[shuffle_indices])

        return np.asarray(batch_samples)


# Helper functions
def weight_variable(shape, name):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class QLearningAgent(bp.Policy):
    def deep_nn(self, X_input):

        # The model
        X_input = tf.reshape(X_input, [-1, INPUT_SIZE])
        Y1 = tf.nn.leaky_relu(tf.matmul(X_input, self.W1) + self.B1)
        Y2 = tf.nn.leaky_relu(tf.matmul(Y1, self.W2) + self.B2)
        Y3 = tf.nn.leaky_relu(tf.matmul(Y2, self.W3) + self.B3)
        Y4 = tf.nn.tanh(tf.matmul(Y3, self.W3) + self.B3)
        Y_logitis = tf.matmul(Y4, self.W_LAST_LAYER) + self.B_LAST_LAYER
        predict = tf.argmax(Y_logitis, 1)

        return Y_logitis, predict

    def init_variables(self):
        try:
            model = pickle.load(open(self.load_from, 'rb'))
            self.W1 = tf.Variable(tf.constant(model[0]))
            self.B1 = tf.Variable(tf.constant(model[1]))
            self.W2 = tf.Variable(tf.constant(model[2]))
            self.B2 = tf.Variable(tf.constant(model[3]))
            self.W3 = tf.Variable(tf.constant(model[4]))
            self.B3 = tf.Variable(tf.constant(model[5]))
            self.W_LAST_LAYER = tf.Variable(tf.constant(model[6]))
            self.B_LAST_LAYER = tf.Variable(tf.constant(model[7]))
        except:
            self.W1 = weight_variable([INPUT_SIZE, FC1], name="W1")
            self.B1 = bias_variable([FC1])
            self.W2 = weight_variable([FC1, FC2], name="W2")
            self.B2 = bias_variable([FC2])
            self.W3 = weight_variable([FC2, FC3], name="W3")
            self.B3 = bias_variable([FC3])
            # W4 = weight_variable([FC3, FC4], name="W4")
            # B4 = bias_variable([FC4])
            self.W_LAST_LAYER = weight_variable([FC3, NUM_ACTIONS], name="W_LAST_LAYER")
            self.B_LAST_LAYER = bias_variable([NUM_ACTIONS])

    def take(self, indices):
        """Return an operation that takes values from network outputs.
        e.g. NN.predict_max() == NN.take(NN.predict_argmax())
        """
        mask = tf.one_hot(indices=indices, depth=NUM_ACTIONS, dtype=tf.bool,
                          on_value=True, off_value=False, axis=-1)
        return tf.boolean_mask(self.output, mask)

    def init_run(self, save_path="test_a1.pkl", folder="/tmp/model_connect_4/",l_rate=LEANING_RATE, session=None, epsilon=0.1):
        self.log("Creating model...layers={}|{}|{},batch={}lr={}".format(FC1, FC2, FC3, BATCH_SIZE, l_rate))
        self.learning_rate = LEANING_RATE
        self.wins = 0
        self.save_to = save_path
        self.model_folder = folder
        self.load_from = "models/" + self.save_to  # not sure about that
        self.epsilon = epsilon

        self.g = tf.Graph()
        with self.g.as_default():
            self.saver = None
            self.input = tf.placeholder(tf.float32, shape=(None, INPUT_SIZE), name="input")
            self.actions = tf.placeholder(tf.int32, (None,), "actions")
            self.init_variables()

            self.output, self.output_argmax = self.deep_nn(self.input)
            self.output_max = tf.reduce_max(self.output, axis=1)
            self.q_values = self.take(self.actions)
            self.q_estimation = tf.placeholder(tf.float32, (None,),
                                               name="q_estimation")
            self.loss = tf.reduce_sum(tf.square(self.q_estimation - self.q_values))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
            self.train_op = self.optimizer.minimize(self.loss)
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
            # self.load(None)

            self.ex_replay = ExperienceReplay()  # ExperienceReplay(*self.memory_args_dict)

    def cast_string_args(self, policy_args):
        # Example
        policy_args['save_to'] = str(policy_args['save_to']) if 'save_to' in policy_args else 'policy_302867833.model.pkl'
        return policy_args

    def predict_max(self, inputs_feed, batch_size=None):
        """Return max on NN outputs."""
        self.output_max = tf.reduce_max(self.output, axis=1)
        out_max, out_argmax = self.session.run([self.output_max, self.output_argmax],
                                               feed_dict={self.input: inputs_feed.reshape(-1,
                                                                                          STATE_DIM)})
        return out_max, out_argmax

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        self.batch_size = BATCH_SIZE
        if prev_state is not None and new_state is not None:
            self.ex_replay.store_last_move([prev_state, prev_action, reward, new_state])
        # self.ex_replay.store([prev_state, prev_action, reward, new_state])

        x_batces_generator = self.ex_replay.get_balanced_batch(batch_size=self.batch_size)
        for batch in x_batces_generator:
            for j, sample in enumerate(batch):
                s1, action, reward, s2 = sample.reshape(4, )
                if s1 is None:
                    break
                arg_v, v = self.predict_max(s2, self.batch_size)

                legal_actions = self.get_legal_moves(s1)
                if action not in legal_actions:
                    self.log("PUNISHED FOR ACTION ={}".format(action))
                    q = -1 + (GAMMA_FACTOR * v)  # penalize for illegal action
                elif reward == 1:  # win or lose the game
                    q = np.asarray([reward])
                else:
                    q = reward + (GAMMA_FACTOR * v)

                feed_dict = {
                    self.input: s1.reshape(-1, STATE_DIM),
                    self.actions: action.reshape(-1, ),
                    self.q_estimation: q.reshape(-1, )
                }
                self.log("legal actions:{}".format(legal_actions))
                self.log("rewards={},q={},v={},actions={}".format(reward, q, v, action))

                # Train on Q'=(s', a') ; s'-new_state, a'-predicted action
                self.session.run(self.train_op, feed_dict=feed_dict)
                if (round + 1) % 5000 == 0:
                    self.epsilon = max(self.epsilon / 2, 1e-4)
                # if round % 30 ==0 and i % 11 == 0:
                self.log("Iteration: {}/{}, round:{}, memory size={}".format(j, self.batch_size, round, self.ex_replay.memory_len))

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        legal_actions = self.get_legal_moves(new_state)
        if self.mode == 'test':
            action = self.session.run(self.output_argmax, feed_dict={self.input: new_state.reshape(-1, STATE_DIM)})[0]
            if action in legal_actions:
                return action
            else:
                return np.random.choice(legal_actions)
        else:
            if prev_state is not None:
                self.ex_replay.store([prev_state, prev_action, reward, new_state])
            action = self.session.run(self.output_argmax, feed_dict={self.input: new_state.reshape(-1, STATE_DIM)})[0]
            if np.random.random() < self.epsilon:
                return np.random.choice(legal_actions)
            else:
                return action

    def load(self, path=None):
        """Load weights or init variables if path==None."""
        # path="/tmp/model_connect_4/"
        if self.saver is None:
            self.saver = tf.train.Saver(max_to_keep=None)

        if path is None or (os.path.exists(path) and not os.listdir(path)):
            self.session.run(tf.global_variables_initializer())
            return 0
        elif path is not None:
            p = Path(path)

            files = p.glob("**/model.ckpt.meta")
            newest = max(files, key=lambda p: p.stat().st_ctime)
            fname = str(newest)[:-5]
            # fname = "./tmp/model.ckpt.meta"
            self.saver.restore(self.session, fname)
            self.log("Model has been loaded successfully, {},{}".format(fname, int(newest.parts[-2])))
            return int(newest.parts[-2])

    def save_model_old(self):
        """Save the current graph."""
        if self.saver is None:
            with self.g.as_default():
                self.saver = tf.train.Saver(max_to_keep=None)

        p = Path(self.save_to)
        p.mkdir(parents=True, exist_ok=True)

        # fname = str(p / "moves_{}_player_{}_layers={}_{}_{}_{}_batch={}lr_{}".format(self.ex_replay.memory_len, self.id, FC1, FC2, FC3, FC4,
        #                                                                              BATCH_SIZE, self.learning_rate) / "model.ckpt")
        fname = str(p / "{}{}".format(self.id, self.ex_replay.memory_len) / "model.ckpt")
        self.saver.save(self.session, fname)
        self.log("Model saved in file: %s" % self.save_to)
        self.log("moves_{}_player_{}_layers={}_{}_{}_batch={}lr_{}".format(self.ex_replay.memory_len, self.id, FC1, FC2, FC3,
                                                                           BATCH_SIZE, self.learning_rate))

        return

    def save_model(self):
        return [self.session.run(self.W1), self.session.run(self.B1),
                self.session.run(self.W2), self.session.run(self.B2),
                self.session.run(self.W3), self.session.run(self.B3),
                self.session.run(self.W_LAST_LAYER), self.session.run(self.B_LAST_LAYER)
                ], self.save_to
