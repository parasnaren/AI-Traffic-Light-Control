""" Deep RL Algorithms for OpenAI Gym environments
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical
import logging
logging.basicConfig(level=logging.INFO)

from ddqn import DDQN
from generator import TrafficGenerator
from utils.networks import get_session
from utils.utils import import_test_configuration, set_sumo

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(args=None):

    config = import_test_configuration(config_file='testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_vehicles_generated']
    )

    input_dim = config['num_states']
    output_dim = config['num_actions']
    model_type = config['model_type']
    episode_seed = config['episode_seed']
    with_per = config.get('with_per')
    dueling = config.get('dueling')
    model_path = config.get('model_path')

    set_session(get_session())
    summary_writer = tf.summary.FileWriter(model_type + "/tensorboard_")

    export_path = 'models/{}'.format(model_type)
    if with_per:
        export_path += '_per'
    if dueling:
        export_path += '_dueling'

    if os.path.exists(export_path):
        export_path = os.path.join(export_path, model_path)
        logging.info('export_path for testing model: {} exists'.format(export_path))

    # Pick algorithm to train
    algo = DDQN(TrafficGen, sumo_cmd, input_dim, output_dim, config, export_path)

    algo.simulate_test(summary_writer, episode_seed=episode_seed)

if __name__ == "__main__":
    main()
