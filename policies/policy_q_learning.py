import os
from pathlib import Path

import numpy as np
import tensorflow as tf

np.random.seed(1231)
from policies import base_policy as bp

LEANING_RATE = 1e-2
BATCH_SIZE = 8
GAMMA_FACTOR = 0.99
NUM_ACTIONS = 7
STATE_DIM = 7 * 6  # board size
INPUT_SIZE = STATE_DIM
FC1 = 64
FC2 = 32
FC3 = 32
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

    def inverse_last_move(self, states, game_over):
        """
        
        :param states: 
        :param game_over: 
        :return: 
        """
        s1, action, reward, s2 = states
        flip_s1 = s1*2 % 3
        flip_s2 = s2*2 % 3
        reward *= -1
        game_over *= -1
        return [flip_s1, action, reward, flip_s2], game_over

    def store_last_move(self, states, game_over):
    def store_last_move(self, states):
        # Save a state to memory, game over = 1 otherwise 0
        self.memory_last_move.append([states, game_over])
        # will learn also the inverse last move
        flip_states, flip_game_over = self.inverse_last_move(states, game_over)
        self.memory_last_move.append([flip_states, flip_game_over])
        self.memory_last_move.append([states])
        # We don't want to store infinite memories, so if we have too many, we just delete the oldest one
        if len(self.memory_last_move) > self.max_memory:
            del self.memory_last_move[:100]

    def get_batch(self, batch_size=32):
        '''
        Here we load one transition <s, a, r, s’> from memory

        :param batch_size:

        :return: a permutation of:
        state_t: prev state s
        action_t: action taken a
        reward_t: reward earned r
        state_tp1: the state that followed s’
        return state_t, action_t, reward_t, state_tp1
        '''
        self.memory_len = len(self.memory)
        shuffle_indices = np.random.permutation(min(self.memory_len, batch_size, int(batch_size)))
        yield np.asarray(self.memory)[shuffle_indices]

    def get_last_move_batch(self, batch_size=32):
        '''
        Here we load one transition <s, a, r, s’> from memory

        :param batch_size:

        :return: a permutation of:
        state_t: prev state s
        action_t: action taken a
        reward_t: reward earned r
        state_tp1: the state that followed s’
        return state_t, action_t, reward_t, state_tp1
        '''
        last_move_len = len(self.memory_last_move)
        last_move_shuffle_indices = np.random.permutation(min(last_move_len, int(batch_size)))
        yield np.asarray(self.memory_last_move)[last_move_shuffle_indices]

    def get_balanced_batch(self, batch_size=32):
        '''
        Get balanced last move samples and regular samples
        :param batch_size:
        :return: 2 batches
        '''
        batch_size /= 2  # divding batch size equally
        batch_samples = []
        last_move_len = len(self.memory_last_move)
        self.memory_len = len(self.memory)
        # sampling batch/2 regular game
        shuffle_indices = np.random.permutation(min(int(batch_size), self.memory_len))
        np.random.shuffle(shuffle_indices)  # get shuffled batch
        batch_samples.append(np.asarray(self.memory)[shuffle_indices])
        # more batched of last state
        for i in range(2):
            # sampling batch/2 last move states
            last_move_shuffle_indices = np.random.permutation(min(last_move_len, int(batch_size)))
            batch_samples.append(np.asarray(self.memory_last_move)[last_move_shuffle_indices])

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


def deep_nn(X_input):
    W1 = weight_variable([INPUT_SIZE, FC1], name="W1")
    B1 = bias_variable([FC1])
    W2 = weight_variable([FC1, FC2], name="W2")
    B2 = bias_variable([FC2])
    W3 = weight_variable([FC2, FC3], name="W3")
    B3 = bias_variable([FC3])
    # W4 = weight_variable([FC3, FC4], name="W4")
    # B4 = bias_variable([FC4])
    W_LAST_LAYER = weight_variable([FC3, NUM_ACTIONS], name="W_LAST_LAYER")
    B_LAST_LAYER = bias_variable([NUM_ACTIONS])

    # The model
    with tf.name_scope('reshape'):
        X_input = tf.reshape(X_input, [-1, INPUT_SIZE])
        Y1 = tf.nn.leaky_relu(tf.matmul(X_input, W1) + B1)
        Y2 = tf.nn.leaky_relu(tf.matmul(Y1, W2) + B2)
        Y3 = tf.nn.leaky_relu(tf.matmul(Y2, W3) + B3)
        Y4 = tf.nn.tanh(tf.matmul(Y3, W3) + B3)
        Y_logitis = tf.matmul(Y4, W_LAST_LAYER) + B_LAST_LAYER
        predict = tf.argmax(Y_logitis, 1)

    return Y_logitis, predict


class QLearningAgent(bp.Policy):
    def take(self, indices):
        """Return an operation that takes values from network outputs.
        e.g. NN.predict_max() == NN.take(NN.predict_argmax())
        """
        mask = tf.one_hot(indices=indices, depth=NUM_ACTIONS, dtype=tf.bool,
                          on_value=True, off_value=False, axis=-1)
        return tf.boolean_mask(self.output, mask)

    def init_run(self, save_path=None, l_rate=LEANING_RATE, session=None):

        self.log("Creating model...layers={}|{}|{},batch={}lr={}".format(FC1, FC2, FC3, BATCH_SIZE, l_rate))
        self.learning_rate = LEANING_RATE
        self.wins = 0

        self.g = tf.Graph()
        with self.g.as_default():
            self.session = tf.Session() if session is None else session
            self.saver = None

            self.input = tf.placeholder(tf.float32, shape=(None, INPUT_SIZE), name="input")
            self.actions = tf.placeholder(tf.int32, (None,), "actions")
            self.output, self.output_argmax = deep_nn(self.input)
            self.output_max = tf.reduce_max(self.output, axis=1)
            self.q_values = self.take(self.actions)
            self.q_estimation = tf.placeholder(tf.float32, (None,),
                                               name="q_estimation")
            self.loss = tf.reduce_mean((self.q_estimation - self.q_values) ** 2)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
            self.train_op = self.optimizer.minimize(self.loss)

            self.load(None)

            self.ex_replay = ExperienceReplay()  # ExperienceReplay(*self.memory_args_dict)

    def cast_string_args(self, policy_args):
        # Example
        policy_args['depth'] = int(policy_args['depth']) if 'depth' in policy_args else 1
        return policy_args

    def predict_max(self, inputs_feed, batch_size=None):
        """Return max on NN outputs."""
        self.output_max = tf.reduce_max(self.output, axis=1)

        return self.session.run(self.output_max, feed_dict={self.input: inputs_feed.reshape(-1, STATE_DIM)})

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        self.batch_size = BATCH_SIZE
        self.ex_replay.store_last_move([prev_state, prev_action, reward, new_state])
        self.ex_replay.store([prev_state, prev_action, reward, new_state])

        x_batces_generator = self.ex_replay.get_balanced_batch(batch_size=self.batch_size)
        # x_batces_last_move = self.ex_replay.get_last_move_batch(batch_size=self.batch_size)
        for j, batch in enumerate(x_batces_generator):
            batch = np.reshape(batch, (-1,4))
            s1_batch = np.asarray([x[0] for x in batch if x[0] is not None])
            action_batch = np.asarray([x[1] for x in batch if x[0] is not None])
            reward_batch = np.asarray([x[2] for x in batch if x[0] is not None])
            s2_batch = np.asarray([x[3] for x in batch if x[0] is not None])
            # s1, action, reward, s2 = sample

            v = self.predict_max(s2_batch, self.batch_size)
            legal_actions = np.array(np.where(s1_batch[0, :] == EMPTY_VAL))
            legal_actions = np.reshape(legal_actions, (legal_actions.size,))

            if v not in legal_actions:
                q = -1 + (GAMMA_FACTOR * v)  # penalize for illegal action
            else:
                q = reward + (GAMMA_FACTOR * v)
            feed_dict = {
                self.input: s1_batch.reshape(-1, STATE_DIM),
                self.actions: action_batch.reshape(-1, ),
                self.q_estimation: q.reshape(-1, )
            }
            self.log("rewards={},q={},v={},actions={}".format( reward_batch,q,v,action_batch))

            # Train on Q'=(s', a') ; s'-new_state, a'-predicted action
            self.session.run(self.train_op, feed_dict=feed_dict)

            # if round % 30 ==0 and i % 11 == 0:
            self.log("Iteration: {}/{}, round:{}, memory size={}".format(j, self.batch_size, round, self.ex_replay.memory_len))

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        legal_actions = np.array(np.where(new_state[0, :] == EMPTY_VAL))
        legal_actions = np.reshape(legal_actions, (legal_actions.size,))

        self.ex_replay.store([prev_state, prev_action, reward, new_state])
        action = self.session.run(self.output_argmax, feed_dict={self.input: new_state.reshape(-1, STATE_DIM)})[0]
        if np.random.random() < self.epsilon:
            return np.random.choice(legal_actions)
        elif action in legal_actions:
            return action
        else:
            return np.random.choice(legal_actions)

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

    def save_model(self, save_path="/tmp/model_connect_4/"):
        """Save the current graph."""
        if self.saver is None:
            with self.g.as_default():
                self.saver = tf.train.Saver(max_to_keep=None)

        p = Path(save_path)
        p.mkdir(parents=True, exist_ok=True)

        # fname = str(p / "moves_{}_player_{}_layers={}_{}_{}_{}_batch={}lr_{}".format(self.ex_replay.memory_len, self.id, FC1, FC2, FC3, FC4,
        #                                                                              BATCH_SIZE, self.learning_rate) / "model.ckpt")
        fname = str(p / "{}{}".format(self.id, self.ex_replay.memory_len) / "model.ckpt")
        self.saver.save(self.session, fname)
        self.log("Model saved in file: %s" % save_path)
        self.log("moves_{}_player_{}_layers={}_{}_{}_batch={}lr_{}".format(self.ex_replay.memory_len, self.id, FC1, FC2, FC3,
                                                                           BATCH_SIZE, self.learning_rate))

        return
