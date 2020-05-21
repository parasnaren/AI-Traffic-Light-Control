""" Training thread for A3C
"""
import traci
import numpy as np
import random
import timeit
import os

from threading import Thread, Lock, get_ident
from keras.utils import to_categorical
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

episode = 0
lock = Lock()
sum_neg_reward = 0
# sum_queue_length = 0
# sum_waiting_time = 0
waiting_times = {}


def training_thread(agent, TrafficGen, total_episodes, sumo_cmd, output_dim, input_dim, training_interval, summary_writer, 
    tqdm, max_steps, green_duration, yellow_duration):
    """ Build threads to run shared computation
    """
    global episode
    while episode < total_episodes:

        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        TrafficGen.generate_routefile(seed=episode)
        sim = 'sim-{}'.format(str(get_ident()))
        traci.start(sumo_cmd, label=sim)
        env = traci.getConnection(sim)

        # Reset episode
        step = 0
        
        old_total_wait = 0
        old_state = _get_state(env, input_dim)
        old_action = -1
        time, cumul_reward = 0, 0
        actions, states, rewards = [], [], []

        while step < max_steps:

            # Actor picks an action (following the policy)
            action = agent.policy_action(np.expand_dims(old_state, axis=0))
            print('action: ', action)

            if step != 0 and old_action != action:
                _set_yellow_phase(env, old_action)
                step = _simulate(env, step, yellow_duration, max_steps)

            # execute the phase selected before
            _set_green_phase(env, action)
            step = _simulate(env, step, green_duration, max_steps)

            current_state = _get_state(env, input_dim)

            current_total_wait = _collect_waiting_times(env)
            reward = old_total_wait - current_total_wait

            # Memorize (s, a, r) for training
            actions.append(to_categorical(action, output_dim))
            rewards.append(reward)
            states.append(old_state)

            # Update current state
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait
            cumul_reward += reward
            time += 1

            # Asynchronous training
            if(time%training_interval==0 or step==max_steps-1):
                lock.acquire()
                agent.train_models(states, actions, rewards)
                lock.release()
                actions, states, rewards = [], [], []

        traci.close(env)

        # Export results for Tensorboard
        score = tfSummary('score', cumul_reward)
        summary_writer.add_summary(score, global_step=episode)
        summary_writer.flush()

        # Update episode count
        with lock:
            tqdm.set_description("Score: " + str(cumul_reward))
            tqdm.update(1)
            if(episode < total_episodes):
                episode += 1

    
def _simulate(env, step, steps_todo, max_steps):
    """
    Execute steps in sumo while gathering statistics
    """
    if (step + steps_todo) >= max_steps:  # do not do more steps than the maximum allowed number of steps
        steps_todo = max_steps - step

    while steps_todo > 0:
        env.simulationStep()  # simulate 1 step in sumo
        step += 1 # update the step counter
        steps_todo -= 1
        queue_length = _get_queue_length(env)
        # sum_queue_length += queue_length
        # sum_waiting_time += queue_length # 1 step while wating in queue means 1 second waited, for each car, therefore queue_lenght == waited_seconds

    return step


def _collect_waiting_times(env):
    """
    Retrieve the waiting time of every car in the incoming roads
    """
    incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
    vehicle_list = env.vehicle.getIDList()
    for vehicle_id in vehicle_list:
        wait_time = env.vehicle.getAccumulatedWaitingTime(vehicle_id)
        road_id = env.vehicle.getRoadID(vehicle_id)  # get the road id where the car is located
        if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
            waiting_times[vehicle_id] = wait_time
        else:
            if vehicle_id in waiting_times: # a car that was tracked has cleared the intersection
                del waiting_times[vehicle_id] 
    total_waiting_time = sum(waiting_times.values())
    return total_waiting_time


def _set_yellow_phase(env, old_action):
    """
    Activate the correct yellow light combination in sumo
    """
    yellow_phase_code = old_action * 2 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
    env.trafficlight.setPhase("TL", yellow_phase_code)


def _set_green_phase(env, action_number):
    """
    Activate the correct green light combination in sumo
    """
    if action_number == 0:
        env.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
    elif action_number == 1:
        env.trafficlight.setPhase("TL", PHASE_WEL_GREEN)
    elif action_number == 2:
        env.trafficlight.setPhase("TL", PHASE_S_EL_GREEN)
    elif action_number == 3:
        env.trafficlight.setPhase("TL", PHASE_W_SL_GREEN)
    elif action_number == 4:
        env.trafficlight.setPhase("TL", PHASE_N_WL_GREEN)
    elif action_number == 5:
        env.trafficlight.setPhase("TL", PHASE_E_NL_GREEN)
        

def _get_queue_length(env):
    """
    Retrieve the number of cars with speed = 0 in every incoming lane
    """
    halt_N = env.edge.getLastStepHaltingNumber("N2TL")
    halt_S = env.edge.getLastStepHaltingNumber("S2TL")
    halt_E = env.edge.getLastStepHaltingNumber("E2TL")
    halt_W = env.edge.getLastStepHaltingNumber("W2TL")
    queue_length = halt_N + halt_S + halt_E + halt_W
    return queue_length


def _get_state(env, input_dim):
    """
    Retrieve the state of the intersection from sumo, in the form of cell occupancy
    """
    state = np.zeros(input_dim)
    vehicle_list = env.vehicle.getIDList()

    for vehicle_id in vehicle_list:
        lane_pos = env.vehicle.getLanePosition(vehicle_id)
        lane_id = env.vehicle.getLaneID(vehicle_id)
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


def _save_episode_stats():
    """
    Save the stats of the episode to plot the graphs at the end of the session
    """
    reward_store.append(sum_neg_reward)  # how much negative reward in this episode
    cumulative_wait_store.append(sum_waiting_time)  # total number of seconds waited by cars in this episode
    avg_queue_length_store.append(sum_queue_length / max_steps)  # average number of queued cars per step, in this episode
