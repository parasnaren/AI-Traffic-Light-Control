import sys
import random
import numpy as np
import traci
import timeit
from random import random, randrange

from tqdm import tqdm
from .agent import Agent
from utils.memory_buffer import MemoryBuffer
from utils.networks import tfSummary

# phase codes based on environment.net.xml
PHASE_NSL_GREEN = 0  # action 0
PHASE_NSL_YELLOW = 1
PHASE_WEL_GREEN = 2  # action 1
PHASE_WEL_YELLOW = 3
PHASE_S_EL_GREEN = 4  # action 2
PHASE_S_EL_YELLOW = 5
PHASE_W_SL_GREEN = 6  # action 3
PHASE_W_SL_YELLOW = 7
PHASE_N_WL_GREEN = 8  # action 4
PHASE_N_WL_YELLOW = 9
PHASE_E_NL_GREEN = 10  # action 5
PHASE_E_NL_YELLOW = 11


class DDQN:
    """ Deep Q-Learning Main Algorithm
    """

    def __init__(self, TrafficGen, sumo_cmd, input_dim, output_dim, config, export_path):
        """ Initialization
        """
        # Environment and DDQN parameters
        self.with_per = config.get('with_per', False)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.export_path = export_path
        
        self.lr = config.get('learning_rate', 0.001)
        self.gamma = config.get('gamma', 0.75)
        self.epsilon = 1.0
        self.training_epochs = config.get('training_epochs', 1)
        self.buffer_size = config.get('buffer_size', 50000)
        self.tau = 1e-2
        self.dueling = config.get('dueling', False)

        self.batch_size = config.get('batch_size', 100)
        self.total_episodes = config.get('total_episodes', 1)
        self.max_steps = config['max_steps']
        self.green_duration = config['green_duration']
        self.yellow_duration = config['yellow_duration']

        self.reward_store = []
        self.cumulative_wait_store = []
        self.avg_queue_length_store = []

        self.sumo_cmd = sumo_cmd

        # Init the traffic generator
        self.TrafficGen = TrafficGen

        # Create actor and critic networks
        self.agent = Agent(self.input_dim, self.output_dim, self.lr, self.tau, self.dueling, config)

        # Memory Buffer for Experience Replay
        self.buffer = MemoryBuffer(self.buffer_size, self.with_per)


    def policy_action(self, state):
        """ Apply an espilon-greedy policy to pick next action
        """
        if random() <= self.epsilon:
            return randrange(self.output_dim)
        else:
            return np.argmax(self.agent.predict(state)[0])


    def train_agent(self):
        """ Train Q-network on batch sampled from the buffer
        """
        # Sample experience from memory buffer (optionally with PER)
        state, action, reward, done, new_state, idx = self.buffer.sample_batch(self.batch_size)

        # Apply Bellman Equation on batch samples to train our DDQN
        q = self.agent.predict(state)
        next_q = self.agent.predict(new_state)
        q_targ = self.agent.target_predict(new_state)

        for i in range(state.shape[0]):
            old_q = q[i, action[i]]
            if done[i]:
                q[i, action[i]] = reward[i]
            else:
                next_best_action = np.argmax(next_q[i,:])
                q[i, action[i]] = reward[i] + self.gamma * q_targ[i, next_best_action]

            if(self.with_per):
                # Update PER Sum Tree
                self.buffer.update(idx[i], abs(old_q - q[i, action[i]]))

        # Train on batch
        self.agent.fit(state, q)


    def simulate(self, summary_writer):
        """ Main DDQN Training Algorithm
        """
        results = []
        tqdm_e = tqdm(range(self.total_episodes), desc='Score', leave=True, unit=" episodes")

        for episode in tqdm_e:

            # Reset episode
            start_time = timeit.default_timer()

            # first, generate the route file for this simulation and set up sumo
            self.TrafficGen.generate_routefile(seed=episode)

            traci.start(self.sumo_cmd)
            print("Simulating...")

            # inits
            self.step = 0
            self.waiting_times = {}
            self.sum_neg_reward = 0
            self.sum_waiting_time = 0
            self.total_delay = 0
            self.queue_length_episode = []

            old_total_wait = 0
            old_state = -1
            old_action = -1
            done = False
            actions, states, rewards = [], [], []

            while self.step < self.max_steps:

                current_state = self.get_state()

                current_total_wait = self.collect_waiting_times()
                self.total_delay += current_total_wait
                reward = old_total_wait - current_total_wait

                # Actor picks an action (following the policy)
                action = self.policy_action(np.expand_dims(old_state, axis=0))

                if self.step != 0 and old_action != action:
                    self.set_yellow_phase(old_action)
                    self.simulateStep(self.yellow_duration)

                # execute the phase selected before
                self.set_green_phase(action)
                self.simulateStep(self.green_duration)

                # Update current state and old action & reward
                old_state = current_state
                old_action = action
                old_total_wait = current_total_wait

                if reward < 0:
                    self.sum_neg_reward += reward

                if self.step >= self.max_steps:
                    done = True

                if self.step != 0:
                    self.memorize(old_state, action, reward, done, current_state)

            self.save_episode_stats()
            traci.close()

            simulation_time = round(timeit.default_timer() - start_time, 1)
            print('Simulation duration:', simulation_time, "/ Epsilon:", round(self.epsilon, 2))

            # Add decay to epsilon
            self.epsilon = 1.0 - (episode / self.total_episodes)

            # Train DDQN and transfer weights to target network
            print("Training...")
            start_time = timeit.default_timer()

            for _ in range(self.training_epochs):
                self.train_agent()
                self.agent.transfer_weights()

            if episode > 0 and episode % 10 == 0:
                self.save_model(self.export_path)

            training_time = round(timeit.default_timer() - start_time, 1)
            print('Training duration:', training_time)

            print('Episode duration: ', simulation_time + training_time)

            # Export results for Tensorboard
            score = tfSummary('score', self.sum_neg_reward)
            summary_writer.add_summary(score, global_step=episode)
            summary_writer.flush()

            # Display score
            tqdm_e.set_description("Reward: " + str(self.sum_neg_reward))
            tqdm_e.refresh()

        return results


    def simulate_test(self, summary_writer, episode_seed):
        """ Main DDQN testing Algorithm
        """
        results = []
        tqdm_e = tqdm(range(1), desc='Score', leave=True, unit=" episodes")

        for episode in tqdm_e:

            # Reset episode
            start_time = timeit.default_timer()

            # first, generate the route file for this simulation and set up sumo
            self.TrafficGen.generate_routefile(seed=episode_seed)

            traci.start(self.sumo_cmd)
            print("Simulating...")

            # inits
            self.step = 0
            self.waiting_times = {}
            self.sum_neg_reward = 0
            self.sum_waiting_time = 0
            self.sum_queue_length = 0
            self.total_delay = 0
            self.queue_length_episode = []

            old_total_wait = 0
            old_state = self.get_state()
            old_action = -1

            while self.step < self.max_steps:

                current_state = self.get_state()

                current_total_wait = self.collect_waiting_times()
                self.total_delay += current_total_wait
                reward = old_total_wait - current_total_wait

                # Actor picks an action (following the policy)
                action = np.argmax(self.agent.predict(np.expand_dims(old_state, axis=0))[0])

                if self.step != 0 and old_action != action:
                    self.set_yellow_phase(old_action)
                    self.simulateStep(self.yellow_duration)

                # execute the phase selected before
                self.set_green_phase(action)
                self.simulateStep(self.green_duration)

                # Update current state and old action & reward
                old_state = current_state
                old_action = action
                old_total_wait = current_total_wait

                if reward < 0:
                    self.sum_neg_reward += reward

            self.save_episode_stats()
            traci.close()
            self.display_episode_stats()
            simulation_time = round(timeit.default_timer() - start_time, 1)
            print('Simulation duration:', simulation_time, "/ Epsilon:", round(self.epsilon, 2))

            # Export results for Tensorboard
            score = tfSummary('score', self.sum_neg_reward)
            summary_writer.add_summary(score, global_step=episode)
            summary_writer.flush()

            # Display score
            tqdm_e.set_description("Reward: " + str(self.sum_neg_reward))
            tqdm_e.refresh()

        return results


    def simulateStep(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        if (self.step + steps_todo) >= self.max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self.max_steps - self.step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self.step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self.get_queue_length()
            self.queue_length_episode.append(queue_length)
            self.sum_waiting_time += queue_length # 1 step while wating in queue means 1 second waited, for each car, therefore queue_lengt == waited_seconds


    def collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        vehicle_list = traci.vehicle.getIDList()
        for vehicle_id in vehicle_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
            road_id = traci.vehicle.getRoadID(vehicle_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self.waiting_times[vehicle_id] = wait_time
            else:
                if vehicle_id in self.waiting_times: # a car that was tracked has cleared the intersection
                    del self.waiting_times[vehicle_id] 
        total_waiting_time = sum(self.waiting_times.values())
        return total_waiting_time


    def set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 2 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase("TL", yellow_phase_code)


    def set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_WEL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_S_EL_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_W_SL_GREEN)
        elif action_number == 4:
            traci.trafficlight.setPhase("TL", PHASE_N_WL_GREEN)
        elif action_number == 5:
            traci.trafficlight.setPhase("TL", PHASE_E_NL_GREEN)
            

    def get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length


    def get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state = np.zeros(self.input_dim)
        vehicle_list = traci.vehicle.getIDList()

        for vehicle_id in vehicle_list:
            lane_pos = traci.vehicle.getLanePosition(vehicle_id)
            lane_id = traci.vehicle.getLaneID(vehicle_id)
            lane_pos = 500 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 500 = max len of a road

            # distance in meters from the traffic light -> mapping into cells
            if lane_pos < 8:
                lane_cell = 0
            elif lane_pos < 16:
                lane_cell = 1
            elif lane_pos < 24:
                lane_cell = 2
            elif lane_pos < 32:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 56:
                lane_cell = 5
            elif lane_pos < 80:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 320:
                lane_cell = 8
            elif lane_pos <= 500:
                lane_cell = 9

            # finding the lane where the vehicle is located 
            lane_id_map = {
                'N2TL_0': 0,
                'N2TL_1': 1,
                'N2TL_2': 2,
                'S2TL_0': 3,
                'S2TL_1': 4,
                'S2TL_2': 5,
                'E2TL_0': 6,
                'E2TL_1': 7,
                'E2TL_2': 8,
                'W2TL_0': 9,
                'W2TL_1': 10,
                'W2TL_2': 11,
            }
            if lane_id in lane_id_map:
                lane_group = lane_id_map[lane_id]
            else:
                lane_group = -1

            if lane_group >= 1 and lane_group <= 11:
                vehicle_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-79
                valid_vehicle = True
            elif lane_group == 0:
                vehicle_position = lane_cell
                valid_vehicle = True
            else:
                valid_vehicle = False  # flag for not detecting cars crossing the intersection or driving away from it

            if valid_vehicle:
                state[vehicle_position] = 1  # write the position of the car vehicle_id in the state array in the form of "cell occupied"

        return state


    def save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self.reward_store.append(self.sum_neg_reward)  # how much negative reward in this episode
        self.cumulative_wait_store.append(self.sum_waiting_time)  # total number of seconds waited by cars in this episode
        self.avg_queue_length_store.append(sum(self.queue_length_episode)/len(self.queue_length_episode))


    def display_episode_stats(self):
        print('Total reward: ', self.sum_neg_reward)
        print('Total waiting time: ', self.sum_waiting_time)
        print('Max queue length: ', max(self.queue_length_episode))
        print('Avg queue length: ', sum(self.queue_length_episode)/len(self.queue_length_episode))


    def memorize(self, state, action, reward, done, new_state):
        """ Store experience in memory buffer
        """
        if(self.with_per):
            q_val = self.agent.predict(np.expand_dims(state, axis=0))
            q_val_t = self.agent.target_predict(np.expand_dims(new_state, axis=0))
            next_best_action = np.argmax(q_val)
            new_val = reward + self.gamma * q_val_t[0, next_best_action]
            td_error = abs(new_val - q_val)[0]
        else:
            td_error = 0
        self.buffer.memorize(state, action, reward, done, new_state, td_error)


    def save_model(self, path):
        path += '-LR_{}'.format(self.lr)
        if(self.with_per):
            path += '_PER'
        self.agent.save_model(path)


    def load_model(self, path):
        self.agent.load_model(path)
    
