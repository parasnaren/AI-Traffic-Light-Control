# AI-Traffic-Light-Control

## Description
-   4 vehicle types: **Bus, Car, Bike and Auto** View SUMO simulation type as 'real world' to view these visualisations. Each vehicle is given a penalty factor in order to simulate the how well/bad these vehicle are driven on India roads.
-   3-way lanes with each lane dedicated to single traffic direction
-   6 traffic light state, equating to **6 actions** to be performed by the agent.
-   Each model has corresponding training and testing configs specified in the **.ini** files

## Models
#### 1.  Deep Q-Networks (DQN)

**Model types:**
- DQN
- DQN + LSTM

**Trained models**
-   **model_1/** is a DQN model trained on 100 episodes
-   **model_2/** is a DQN model trained on 300 episodes
-   **model_3/** is a DQN+LSTM model trained on 100 episodes
-   **model_4/** is a DQN+LSTM model trained on 300 episodes


#### 2. Double Deep Q-Network (DDQN)

**Model Types:**
- DDQN
- DDQN + PER *(Prioritized Experience Replay)*
- Dueling DQN (+ PER)


## Setup

1. Download SUMO from [here](https://sumo.dlr.de/docs/Downloads.php) and install the program.
2. Ensure that you set your **PYTHONPATH** variable to point to your SUMO installation folder **/your/path/to/Sumo/tools**
3. Clone the repository.
4. Run `training_main.py` under either the DQN or DDQN directories.
5. Additionally to view the models generated from training, set `gui=True` in the `testing_settings.ini` in order to view the simulation of traffic control.
