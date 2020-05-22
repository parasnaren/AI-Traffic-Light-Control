""" Deep RL Algorithms for OpenAI Gym environments
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical

# from A2C.a2c import A2C
from A3C.a3c import A3C
from generator import TrafficGenerator
from utils.networks import get_session
from utils.utils import import_train_configuration, set_sumo, set_train_path


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main(args=None):

    config = import_train_configuration(config_file='training_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_vehicles_generated']
    )

    epochs = config['training_epochs']
    model_type = config['model_type']
    num_layers = config['num_layers']
    width = config['width_layers']
    batch_size = config['batch_size']
    gamma = config['gamma']
    lr = config['learning_rate']
    input_dim = config['num_states']
    output_dim = config['num_actions']
    n_threads = config['n_threads']
    total_episodes = config['total_episodes']
    max_steps = config['max_steps']
    green_duration = config['green_duration']
    yellow_duration = config['yellow_duration']

    set_session(get_session())
    summary_writer = tf.summary.FileWriter(model_type + "/tensorboard_")

    # Pick algorithm to train
    if(model_type.upper()=="A2C"):
        algo = A3C(TrafficGen, num_layers, width, batch_size, gamma, lr, input_dim, output_dim, 
            n_threads, sumo_cmd, total_episodes, max_steps, green_duration, yellow_duration, epochs)
    elif(model_type.upper()=="A3C"):
        algo = A3C(TrafficGen, num_layers, width, batch_size, gamma, lr, input_dim, output_dim, 
            n_threads, sumo_cmd, total_episodes, max_steps, green_duration, yellow_duration, epochs)

    # Train
    stats = algo.train(summary_writer)

    # Export results to CSV
    # df = pd.DataFrame(np.array(stats))
    # df.to_csv(args.type + "/logs.csv", header=['Episode', 'Mean', 'Stddev'], float_format='%10.5f')

    # Save weights and close environments
    exp_dir = '{}/models/'.format(model_type)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    export_path = '{}_{}_EP_{}_BS_{}'.format(
        exp_dir,
        model_type,
        total_episodes,
        batch_size)

    algo.save_weights(export_path)

if __name__ == "__main__":
    main()
