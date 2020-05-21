import sys
import time
import threading
import numpy as np
import traci

from tqdm import tqdm
from keras.models import Model, Sequential
from keras import regularizers
from keras.layers import Dense

from .critic import Critic
from .actor import Actor
from .thread import training_thread
# from utils.networks import conv_block
# from utils.stats import gather_stats

class A3C:
    """ Asynchronous Actor-Critic Main Algorithm
    """

    def __init__(self, TrafficGen, num_layers, width, batch_size, gamma, lr, input_dim, 
        output_dim, n_threads, sumo_cmd, total_episodes, max_steps, green_duration, yellow_duration, epochs):
        """ Initialization
        """
        # Environment and A3C parameters
        self.TrafficGen = TrafficGen
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.gamma = gamma
        self.lr = lr
        self.epochs = epochs

        # Create actor and critic networks
        self.shared = self.buildNetwork(num_layers, width)
        self.actor = Actor(self.input_dim, output_dim, self.shared, lr, epochs)
        self.critic = Critic(self.input_dim, output_dim, self.shared, lr, epochs)

        # Build optimizers
        self.a_opt = self.actor.optimizer()
        self.c_opt = self.critic.optimizer()

        self.n_threads = n_threads
        self.sumo_cmd = sumo_cmd
        self.total_episodes = total_episodes
        self.max_steps = max_steps
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration


    def buildNetwork(self, num_layers, width):
        """ Assemble shared layers
        """
        model = Sequential()
        model.add(Dense(width, input_shape=(self.input_dim,)))
        for _ in range(num_layers):
            model.add(Dense(width, activation='relu'))
        return model

    def policy_action(self, state):
        """ Use the actor's network to predict the next action to take, using the policy
        """
        print('actor prediction: ', self.actor.predict(state).ravel())
        policy_action = np.random.choice(np.arange(self.output_dim), 1, p=self.actor.predict(state).ravel())[0]
        print('policy_action: ', policy_action)
        return policy_action

    def discount(self, r, s):
        """ Compute the gamma-discounted rewards over an episode
        """
        discounted_r, cumul_r = np.zeros_like(r), 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r

    def train_models(self, states, actions, rewards):
        """ Update actor and critic networks from experience
        """
        # Compute discounted rewards and Advantage (TD. Error)
        discounted_rewards = self.discount(rewards, states[-1])
        state_values = self.critic.predict(np.array(states))
        advantages = discounted_rewards - np.reshape(state_values, len(state_values))
        # Networks optimization
        self.a_opt([states, actions, advantages])
        self.c_opt([states, discounted_rewards])

    def train(self, summary_writer):

        # Instantiate one environment per thread
        # envs = []
        # for i in range(self.n_threads):
        #     sim = 'sim-{}'.format(str(i))
        #     traci.start(self.sumo_cmd, label=sim)
        #     envs.append(traci.getConnection(sim))

        # Create threads
        tqdm_e = tqdm(range(int(self.total_episodes)), desc='Score', leave=True, unit=" episodes")

        threads = [threading.Thread(
                target=training_thread,
                daemon=True,
                args=(self,
                    self.TrafficGen,
                    self.total_episodes,
                    self.sumo_cmd,
                    self.output_dim,
                    self.input_dim,
                    30,
                    summary_writer,
                    tqdm_e,
                    self.max_steps,
                    self.green_duration,
                    self.yellow_duration)) for i in range(self.n_threads)]

        for t in threads:
            t.start()
            time.sleep(0.5)
        try:
            [t.join() for t in threads]
        except KeyboardInterrupt:
            print("Exiting all threads...")
        return None

    def save_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        self.actor.save(path)
        self.critic.save(path)

    def load_weights(self, path_actor, path_critic):
        self.critic.load_weights(path_critic)
        self.actor.load_weights(path_actor)
