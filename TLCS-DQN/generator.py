import numpy as np
import math

class TrafficGenerator:
    def __init__(self, max_steps, n_vehicles_generated):
        self._n_vehicles_generated = n_vehicles_generated  # how many cars per episode
        self._max_steps = max_steps

    def generate_routefile(self, seed):
        """
        Generation of the route of every car for one episode
        """
        np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_vehicles_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        with open("buffer/routes.rou.xml", "w") as routes:

            route_default_string = """
            <routes>
                <vType guiShape="passenger/hatchback" accel="1.0" decel="4.0" id="Car" color="white" width="2.5" length="4.0" minGap="1.0" maxSpeed="80.0" sigma="0.2"/>
                <vType guiShape="bus" accel="0.5" decel="5.0" id="Bus" color="blue" width="3.5" length="6.0" minGap="1.0" maxSpeed="40.0" sigma="0.3"/>
                <vType guiShape="motorcycle" accel="1.0" decel="4.0" id="Bike" color="green" width="1.5" length="2.0" minGap="1.0" maxSpeed="60.0" sigma="0.6"/>
                <vType guiShape="evehicle" accel="1.0" decel="4.0" id="Auto" color="yellow" width="2.0" length="3.0" minGap="1.0" maxSpeed="50.0" sigma="0.6"/>
                
                <route id="W_N" edges="W2TL TL2N"/>
                <route id="W_E" edges="W2TL TL2E"/>
                <route id="W_S" edges="W2TL TL2S"/>
                <route id="N_W" edges="N2TL TL2W"/>
                <route id="N_E" edges="N2TL TL2E"/>
                <route id="N_S" edges="N2TL TL2S"/>
                <route id="E_W" edges="E2TL TL2W"/>
                <route id="E_N" edges="E2TL TL2N"/>
                <route id="E_S" edges="E2TL TL2S"/>
                <route id="S_W" edges="S2TL TL2W"/>
                <route id="S_N" edges="S2TL TL2N"/>
                <route id="S_E" edges="S2TL TL2E"/>"""

            print(route_default_string, file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                vehicle_type = np.random.uniform()
                if vehicle_type < 0.25:
                    vehicle = 'Car'
                elif 0.25 <= vehicle_type < 0.32:
                    vehicle = 'Bus'
                elif 0.32 <= vehicle_type < 0.85:
                    vehicle = 'Bike'
                else:
                    vehicle = 'Auto'

                straight_or_turn = np.random.uniform()
                if straight_or_turn <= 0.4:
                    route_straight = np.random.randint(1, 5)  # choose a random source & destination
                    if route_straight == 1:
                        print('    <vehicle id="W_E_%i" type="%s" route="W_E" depart="%s" departLane="random" departSpeed="10" />' 
                            % (car_counter, vehicle, step), file=routes)
                    elif route_straight == 2:
                        print('    <vehicle id="E_W_%i" type="%s" route="E_W" depart="%s" departLane="random" departSpeed="10" />'
                            % (car_counter, vehicle, step), file=routes)
                    elif route_straight == 3:
                        print('    <vehicle id="N_S_%i" type="%s" route="N_S" depart="%s" departLane="random" departSpeed="10" />'
                            % (car_counter, vehicle, step), file=routes)
                    else:
                        print('    <vehicle id="S_N_%i" type="%s" route="S_N" depart="%s" departLane="random" departSpeed="10" />' 
                            % (car_counter, vehicle, step), file=routes)
                else:
                    route_direction = np.random.uniform()
                    route_turn = np.random.randint(1, 5)
                    if route_direction <= 0.80:
                        if route_turn == 1:
                            print('    <vehicle id="W_N_%i" type="%s" route="W_N" depart="%s" departLane="random" departSpeed="10" />' 
                                % (car_counter, vehicle, step), file=routes)
                        elif route_turn == 2:
                            print('    <vehicle id="W_S_%i" type="%s" route="E_S" depart="%s" departLane="random" departSpeed="10" />' 
                                % (car_counter, vehicle, step), file=routes)
                        elif route_turn == 3:
                            print('    <vehicle id="N_W_%i" type="%s" route="S_W" depart="%s" departLane="random" departSpeed="10" />' 
                                % (car_counter, vehicle, step), file=routes)
                        else:
                            print('    <vehicle id="N_E_%i" type="%s" route="N_E" depart="%s" departLane="random" departSpeed="10" />'
                                % (car_counter, vehicle, step), file=routes)
                    else:
                        if route_turn == 1:
                            print('    <vehicle id="E_N_%i" type="%s" route="E_N" depart="%s" departLane="random" departSpeed="10" />' 
                                % (car_counter, vehicle, step), file=routes)
                        elif route_turn == 2:
                            print('    <vehicle id="E_S_%i" type="%s" route="W_S" depart="%s" departLane="random" departSpeed="10" />' 
                                % (car_counter, vehicle, step), file=routes)
                        elif route_turn == 3:
                            print('    <vehicle id="S_W_%i" type="%s" route="N_W" depart="%s" departLane="random" departSpeed="10" />' 
                                % (car_counter, vehicle, step), file=routes)
                        else:
                            print('    <vehicle id="S_E_%i" type="%s" route="S_E" depart="%s" departLane="random" departSpeed="10" />' 
                                % (car_counter, vehicle, step), file=routes)

            print("</routes>", file=routes)
