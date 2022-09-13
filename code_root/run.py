
from Environment.EmpiricalTravelModel import EmpiricalTravelModel
from decision_making.coordinator.DoNothing import DoNothing as Coord_DoNothing
from decision_making.dispatch.DoNothing import DoNothing as Dispatch_DoNothing
from Environment.Simulator import Simulator
from Environment.EnvironmentModel import EnvironmentModel
import argparse
from src.utils import *
import json
import copy
import spdlog as spd

def load_initial_state():
    return None
    
def load_events():
    return []

if __name__ == '__main__':
    spd.FileLogger(name='test', filename='spdlog_example.log', truncate=True)
    logger = spd.get('test')
    logger.set_pattern("[%l] %v")

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log_level', type=str, default='INFO')
    args = parser.parse_args()
    args = dotdict(namespace_to_dict(args))
    if args.log_level == 'INFO':
        logger.set_level(spd.LogLevel.INFO)
    elif args.log_level == 'DEBUG':
        logger.set_level(spd.LogLevel.DEBUG)
    elif args.log_level == 'ERROR':
        logger.set_level(spd.LogLevel.ERR)
        
    config_path = 'scenarios/baseline/data/config.json'
    with open(config_path) as f:
        config = dotdict(json.load(f))
        
    config_path = f'scenarios/baseline/data/{config.trip_plan}'
    with open(config_path) as f:
        trip_plan = dotdict(json.load(f))
        
    config_path = f'scenarios/baseline/data/{config.vehicle_plan}'
    with open(config_path) as f:
        bus_plan = dotdict(json.load(f))
        
    print(trip_plan['246346'])
    travel_model = EmpiricalTravelModel()
    sim_environment = EnvironmentModel(travel_model)
    do_nothing_dispatcher = Dispatch_DoNothing(travel_model)
    do_nothing_coordinator = Coord_DoNothing(sim_environment, travel_model, do_nothing_dispatcher)
    
    incident_events = load_events()
    starting_state = load_initial_state()
    
    simulator = Simulator(starting_event_queue=copy.deepcopy(incident_events),
                          starting_state=starting_state,
                          environment_model=sim_environment,
                          event_processing_callback=do_nothing_coordinator.event_processing_callback_funct,
                          logger=logger)

    simulator.run_simulation()