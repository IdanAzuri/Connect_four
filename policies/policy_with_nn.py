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
ROWS = 6
COLS = 7
WIN_MASK = np.ones(4)


# Helper functions
def weight_variable(shape, name):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class ExperienceReplay(object):
    """
    During gameplay all the experiences < s, a, r, s’ > are stored in a replay memory. 
    In training, batches of randomly drawn experiences are used to generate the input and target for training.
    """

    def __init__(self, session, x_placeholder, y_placeholder, y_logits, max_memory=100, discount=GAMMA_FACTOR):
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
        self.x_input_ = x_placeholder
        self.y_ = y_placeholder
        self.session_ = session
        self.y_logit_ = y_logits
        self.max_memory = 1e7
        self.memory = []
        self.discount = GAMMA_FACTOR

    def store(self, states, game_over):
        # Save a state to memory, game over = 1 otherwise 0
        self.memory.append([states, game_over])
        # We don't want to store infinite memories, so if we have too many, we just delete the oldest one
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch_and_target(self, batch_size=10):

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
        # TODO FIX SIZE BATCH_SIZE
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
            prev_action_predicted, targets[i] = self.session_.run(self.y_logit_, self.y_, feed_dic={
                self.x_input_: state_t.reshape(
                    -1, STATE_DIM), self.y_: np.ones((1, 7))})[0]

            """
            If the game ended, the expected reward Q(s,a) should be the final reward r.
            Otherwise the target value is r + gamma * max Q(s’,a’)
            """
            #  Here Q_sa is max_a'Q(s', a')
            Q_sa = np.max(prev_action_predicted)

            # if the game ended, the reward is the final reward
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # r + gamma * max Q(s’,a’)
                targets[i, action_t] = reward_t + GAMMA_FACTOR * Q_sa
        return inputs, targets

    def get_batch(self, batch_size=64):
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
        memory_len = len(self.memory)
        # indices_permutation = np.random.permutation(memory_len)[:min(batch_size,memory_len)]
        shuffle_indices = np.arange(memory_len)
        np.random.shuffle(shuffle_indices)  # get shuffled batch
        return np.asarray(self.memory)[shuffle_indices]


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
            self.load()
            self.memory_args_dict = {'session': self.session, 'x_placeholder': self.x_input, 'y_placeholder': self.y,
                                     'y_logits': self.y_logitis}

        self.ex_replay = ExperienceReplay(*self.memory_args_dict)
        # self.session.run(tf.global_variables_initializer())

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        # find legal actions:

        legal_actions = np.array(np.where(new_state[0, :] == EMPTY_VAL))
        legal_actions = np.reshape(legal_actions, (legal_actions.size,))
        legal_actions_hot_vector = np.zeros((7,), dtype=np.int32)
        legal_actions_hot_vector[legal_actions] = 1
        self.batch_size = 64
        self.ex_replay.store([prev_state, prev_action, reward, new_state], int(reward))
        x_batch = self.ex_replay.get_batch(batch_size=self.batch_size)
        for i, v in enumerate(x_batch):
            prev_state, prev_action, reward, new_state = v[0]
            if prev_state is None:
                break
            self.log("Iteration: {}/{}, round:{}".format(i, self.batch_size, round))
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
                if int(
                        prev_action_predicted) in legal_actions:  # if the action is ilegal learn the real game
                    # action and
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
                                                                      self.x_input:
                                                                          state_after_predicted_action.reshape(
                                                                              -1,
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
                #         reward_for_predicted_action + max_action_prob_after_playing) - np.max(
                # predicted_actions_prob_vec)
                # Train our network using target and predicted Q values
                self.session.run([self.trainer, self.loss],
                                 feed_dict={self.x_input: new_state.reshape(-1, INPUT_SIZE),
                                            self.y: predicted_actions_prob_vec})
                # TODO Learnig from real game parameters, combine it to a dictionary to feed the model once
                # Learning wins
                # if reward == 1:
                #     real_prob_vector =np.zeros((1,7))
                #     real_prob_vector[prev_action] = 1  # set the win in 1 hot vector
                #     self.session.run([self.trainer, self.loss],
                #                      feed_dict={self.x_input: new_state.reshape(-1, INPUT_SIZE),
                #                                 self.y: real_prob_vector})
                all_rewards += reward_for_predicted_action
                new_state = state_after_predicted_action

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
        self.ex_replay.store([prev_state, prev_action, reward, new_state], int(reward))
        # TODO maybe to change y argument (tmp_actions)
        action = \
            self.session.run(self.y_argmax, feed_dict={self.x_input: new_state.reshape(-1, STATE_DIM),
                                                       self.y: temp_actions})[0]

        if action in legal_actions:  # and np.random.random() > self.epsilon:
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
