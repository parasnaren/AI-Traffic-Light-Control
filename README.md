# AI-Traffic-Light-Control

## Description
-   4 vehicle types: **Bus, Car, Bike and Auto** View SUMO simulation type as 'real world' to view these visualisations. Each vehicle is given a penalty factor in order to simulate the how well/bad these vehicle are driven on India roads.
-   3-way lanes with each lane dedicated to single traffic direction
-   6 traffic light state, equating to **6 actions** to be performed by the agent.
-   Each model has corresponding training and testing configs specified in the **.ini** files

## Models
### 1.  Deep Q-learning using Neural Network (TLCS-DQN)

**Model types:**
- DQN
- DQN + LSTM

**Trained models**
-   **model_1/** is a DQN model trained on 100 episodes
-   **model_2/** is a DQN model trained on 300 episodes
-   **model_3/** is a DQN+LSTM model trained on 100 episodes
-   **model_4/** is a DQN+LSTM model trained on 300 episodes


### 2. Double Deep Q-learning with Neural Network (TLCS-DDQN)

**Model Types:**
- DDQN
- DDQN + PER *(Prioritized Experience Replay)*
- DDQN + PER + Dueling
