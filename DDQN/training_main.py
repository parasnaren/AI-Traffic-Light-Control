""" Deep RL Algorithms for OpenAI Gym environments
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical

from ddqn import DDQN
from generator import TrafficGenerator
from utils.networks import get_session
from utils.utils import import_train_configuration, set_sumo

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(args=None):

    config = import_train_configuration(config_file='training_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_vehicles_generated']
    )

    input_dim = config['num_states']
    output_dim = config['num_actions']
    model_type = config['model_type']
    total_episodes = config['total_episodes']
    batch_size = config['batch_size']

    set_session(get_session())
    summary_writer = tf.summary.FileWriter(model_type + "/tensorboard_")

    # Save weights and close environments
    exp_path = model_type
    exp_path += '_per' if config['with_per'] else ''
    exp_path += '_dueling' if config['dueling'] else ''
    exp_dir = 'models/{}'.format(exp_path)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        
    export_path = '{}/{}-episode_{}-batch_{}'.format(
        exp_dir,
        model_type,
        total_episodes,
        batch_size)

    # Pick algorithm to train
    if(model_type.upper()=="DDQN"):
        algo = DDQN(TrafficGen, sumo_cmd, input_dim, output_dim, config, export_path)
    elif(model_type.upper()=="DDQN+CLIP"):
        algo = DDQN(TrafficGen, sumo_cmd, input_dim, output_dim, config, export_path)

    # Train
    stats = algo.simulate(summary_writer)

    # Export results to CSV
    # df = pd.DataFrame(np.array(stats))
    # df.to_csv(args.type + "/logs.csv", header=['Episode', 'Mean', 'Stddev'], float_format='%10.5f')

    algo.save_model(export_path)

if __name__ == "__main__":
    main()
