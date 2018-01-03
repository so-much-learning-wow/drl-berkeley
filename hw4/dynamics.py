import gym
import random
import numpy as np
import tensorflow as tf


class NNDynamicsModel():
    def __init__(self,
                 env,
                 n_layers,
                 n_hid_units,
                 activation,
                 output_activation,
                 normalization_stats,
                 batch_size,
                 num_iter,
                 learning_rate,
                 sess
                ):
        """ Note: Be careful about normalization """

        self.normalization_stats = normalization_stats
        self.batch_size = batch_size
        self.num_iter = num_iter
        self.sess = sess

        is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.n if is_discrete else env.action_space.shape[0]

        self._inputs = tf.placeholder(shape=[None, ac_dim + ob_dim], dtype=tf.float32, name='inputs')
        self._targets = tf.placeholder(shape=[None, ob_dim], dtype=tf.float32, name='targets')

        self._model_outputs = build_mlp(self._inputs, ob_dim, 'dynamics_model', activation=activation,
                                        n_hid_units=n_hid_units, n_layers=n_layers, output_activation=output_activation)

        # print(self._targets.shape.as_list(), self._model_outputs.shape.as_list())

        self._loss = tf.losses.mean_squared_error(self._targets, self._model_outputs)

        self._train_op = tf.train.AdamOptimizer(learning_rate).minimize(self._loss)

    def fit(self, paths):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions,
        (unnormalized) next_states and fit the dynamics model going from normalized states,
        normalized actions to normalized state differences (s_t+1 - s_t)
        """

        for i in range(self.num_iter):
            print('Making {}/{} dynamics model training iterations'.format(i+1, self.num_iter), end='. ')

            train_ids = set(random.sample(list(range(len(paths))), 4 * len(paths)//5))
            train_paths = [paths[i] for i in train_ids]
            val_paths = [paths[i] for i in range(len(paths)) if not i in train_ids]

            train_inputs, train_targets = build_dataset(train_paths, self.normalization_stats)
            val_inputs, val_targets = build_dataset(val_paths, self.normalization_stats)

            train_feed_dict = {self._inputs: train_inputs, self._targets: train_targets}
            val_feed_dict = {self._inputs: val_inputs, self._targets: val_targets}

            train_loss, _ = self.sess.run([self._loss, self._train_op], feed_dict=train_feed_dict)
            val_loss = self.sess.run(self._loss, feed_dict=val_feed_dict)

            print('Train loss: {:.2f}. Validation loss: {:.2f}'.format(train_loss, val_loss))


    def predict(self, obs, acs):
        """
        Write a function to take in a batch of (unnormalized) states and
        (unnormalized) actions and return the (unnormalized) next states as
        predicted by using the model
        """

        norm_obs = (obs - self.normalization_stats['obs_mean']) / self.normalization_stats['obs_std']
        norm_acs = (acs - self.normalization_stats['acs_mean']) / self.normalization_stats['acs_std']

        norm_deltas = self.sess.run(self._model_outputs, feed_dict={self._inputs: np.column_stack([norm_obs, norm_acs])})
        deltas = norm_deltas * self.normalization_stats['deltas_std'] + self.normalization_stats['deltas_mean']

        # print(np.min(obs), np.max(obs), np.min(acs), np.max(acs), np.min(deltas), np.max(deltas))

        return obs + deltas


    def multi_step_validation(self, paths):
        """
            Computes multi-step validation loss on the given paths.
            Check section IV.B of the corresponding paper for more info.
        """
        for path in paths:
            pass




# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder,
              output_size,
              scope,
              n_layers=2,
              n_hid_units=500,
              activation=tf.tanh,
              output_activation=None
              ):

    out = input_placeholder

    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, n_hid_units, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)

    return out


# Building train dataset from paths list
def build_dataset(paths, normalization_stats):
    obs = np.concatenate([path['observations'] for path in paths])
    acs = np.concatenate([path['actions'] for path in paths])
    next_obs = np.concatenate([path['next_observations'] for path in paths])
    deltas = next_obs - obs

    norm_obs = (obs - normalization_stats['obs_mean']) / normalization_stats['obs_std']
    norm_acs = (acs - normalization_stats['acs_mean']) / normalization_stats['acs_std']
    norm_deltas = (deltas - normalization_stats['deltas_mean']) / normalization_stats['deltas_std']

    return np.column_stack([norm_obs, norm_acs]), norm_deltas
