import traci
import numpy as np
import random
import timeit
import os

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


class Simulation:
    def __init__(self, Model, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, num_states, num_actions):
        self._Model = Model
        self._TrafficGen = TrafficGen
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_episode = []
        self._queue_length_episode = []


    def run(self, episode):
        """
        Runs the testing simulation
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        old_total_wait = 0
        old_action = -1 # dummy init

        while self._step < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_action = action
            old_total_wait = current_total_wait

            self._reward_episode.append(reward)

        #print("Total reward:", np.sum(self._reward_episode))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time


    def _simulate(self, steps_todo):
        """
        Proceed with the simulation in sumo
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length() 
            self._queue_length_episode.append(queue_length)


    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        vehicle_list = traci.vehicle.getIDList()
        for vehicle_id in vehicle_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
            road_id = traci.vehicle.getRoadID(vehicle_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[vehicle_id] = wait_time
            else:
                if vehicle_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[vehicle_id] 
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time


    def _choose_action(self, state):
        """
        Pick the best action known based on the current state of the env
        """
        action = np.argmax(self._Model.predict_one(state))
        # print('action taken: ', action)
        return action


    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 2 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase("TL", yellow_phase_code)


    def _set_green_phase(self, action_number):
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


    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length


    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state = np.zeros(self._num_states)
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


    @property
    def queue_length_episode(self):
        return self._queue_length_episode


    @property
    def reward_episode(self):
        return self._reward_episode



