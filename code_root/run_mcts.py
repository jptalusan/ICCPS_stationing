# All dates and times should just be datetime!
from decision_making.coordinator.DecisionMaker import DecisionMaker
from decision_making.coordinator.RandomCoord import RandomCoord
from decision_making.dispatch.RandomDispatch import RandomDispatch
from decision_making.dispatch.SendNearestDispatchPolicy import SendNearestDispatchPolicy
from decision_making.DecisionEnvironmentDynamics import DecisionEnvironmentDynamics
from decision_making.CentralizedMCTS.ModularMCTS import ModularMCTS
from decision_making.CentralizedMCTS.Rollout import BareMinimumRollout
from decision_making.ValidActions import ValidActions
from Environment.DataStructures.Bus import Bus
from Environment.DataStructures.Event import Event
from Environment.DataStructures.State import State
from Environment.DataStructures.Stop import Stop
# from Environment.EmpiricalTravelModel import EmpiricalTravelModel
from Environment.EmpiricalTravelModelLookup import EmpiricalTravelModelLookup
from Environment.enums import BusStatus, BusType, EventType, MCTSType
from Environment.EnvironmentModel import EnvironmentModel
from Environment.EnvironmentModelFast import EnvironmentModelFast
from Environment.Simulator import Simulator
from src.utils import *
from tqdm import tqdm
import argparse
import copy
import json
import os
import pickle
import numpy as np
import pandas as pd
import spdlog as spd
import datetime as dt
import sys


def load_initial_state(starting_date, bus_plan, trip_plan, random_seed=100):
    print("Loading initial states...")
    active_stops = []

    _starting_date_str = dt.datetime.strptime(starting_date, '%Y%m%d').strftime('%Y-%m-%d')
    Buses = {}
    for bus_id, bus_info in bus_plan.items():
        bus_type = bus_info['service_type']
        if bus_type == 'regular':
            bus_type = BusType.REGULAR
        else:
            bus_type = BusType.OVERLOAD

        bus_status = BusStatus.IDLE
        bus_capacity = bus_info['vehicle_capacity']
        bus_starting_depot = bus_info['starting_depot']
        bus_block_trips = np.asarray(bus_info['trips'])

        bus_block_trips = [tuple(bus_block_trip) for bus_block_trip in bus_block_trips]
        for i, bus_block_trip in enumerate(bus_block_trips):
            block_id = bus_block_trip[0]
            trip_id = bus_block_trip[1]
            trip_info = trip_plan[trip_id]
            stop_id_original = trip_info['stop_id_original']
            active_stops.extend(stop_id_original)
            if i == 0:
                st = trip_plan[trip_id]['scheduled_time']
                st = [str_timestamp_to_datetime(st).time().strftime('%H:%M:%S') for st in st][0]
                # Add when the bus should reach next stop as state change
                t_state_change = str_timestamp_to_datetime(f"{_starting_date_str} {st}")

        bus = Bus(bus_id,
                  bus_type,
                  bus_status,
                  bus_capacity,
                  bus_block_trips)
        bus.current_stop = bus_starting_depot
        bus.current_load = 0
        bus.t_state_change = t_state_change
        Buses[bus_id] = bus

    Stops = {}
    for active_stop in active_stops:
        stop = Stop(stop_id=active_stop)
        Stops[active_stop] = stop

    print(f"Added {len(Buses)} buses and {len(Stops)} stops.")
    return Buses, Stops


def load_events(travel_model, starting_date, Buses, Stops, trip_plan, event_file="", random_seed=100):
    print("Adding events...")
    np.random.seed(random_seed)
    has_broken = False
    is_weekend = 0 if dt.datetime.strptime(starting_date, '%Y%m%d').weekday() < 5 else 1
    # Load distributions
    with open(f'scenarios/baseline/data/sampled_ons_offs_dict_{starting_date}.pkl', 'rb') as handle:
        sampled_travel_time = pickle.load(handle)

    # Initial events
    # Includes: Trip starts, passenger sampling
    # all active stops that buses will pass
    events = []
    saved_events = f'scenarios/baseline/data/{event_file}'

    _starting_date_str = dt.datetime.strptime(starting_date, '%Y%m%d').strftime('%Y-%m-%d')
    
    pbar = tqdm(Buses.items())
    # if not os.path.exists(saved_events):
    if True:
        for bus_id, bus in pbar:
            if bus.type == BusType.OVERLOAD:
                continue
            blocks_trips = bus.bus_block_trips

            # Start trip (assuming trips are in sequential order)
            block = blocks_trips[0][0]
            trip = blocks_trips[0][1]
            st = trip_plan[trip]['scheduled_time']
            st = [str_timestamp_to_datetime(st).time().strftime('%H:%M:%S') for st in st][0]
            first_stop_scheduled_time = str_timestamp_to_datetime(f"{_starting_date_str} {st}")
            current_block_trip = bus.bus_block_trips.pop(0)
            current_stop_number = bus.current_stop_number
            current_depot = bus.current_stop
            
            bus.current_block_trip = current_block_trip
            bus.current_stop_number = current_stop_number
            
            travel_time, distance = travel_model.get_traveltime_distance_from_depot(current_block_trip,
                                                                                    current_depot,
                                                                                    bus.current_stop_number)
            
            time_to_state_change = first_stop_scheduled_time + dt.timedelta(seconds=travel_time)
            bus.t_state_change = time_to_state_change
            bus.distance_to_next_stop = distance
            bus.status = BusStatus.IN_TRANSIT
            
            event = Event(event_type=EventType.VEHICLE_ARRIVE_AT_STOP,
                               time=time_to_state_change,
                               type_specific_information={'bus_id': bus_id,
                                                          'current_block_trip': current_block_trip,
                                                          'stop': bus.current_stop_number})
            events.append(event)

        events.sort(key=lambda x: x.time, reverse=False)

        # with open(saved_events, "wb") as f:
        #     pickle.dump(events, f)
    else:
        print("loading events...")
        with open(saved_events, "rb") as f:
            events = pickle.load(f)
    
    return events

def manually_insert_disruption(events, buses, bus_id, time):
    if bus_id not in list(buses.keys()):
        raise "Bus does not exist."

    start_time = events[0].time
    end_time = events[-1].time

    # if time < start_time or time > end_time:
    #     raise "Time beyond limits."

    if time < start_time:
        raise "Chosen time is in the past."

    event = Event(event_type=EventType.VEHICLE_BREAKDOWN,
                  time=time,
                  type_specific_information={'bus_id': bus_id})
    events.append(event)
    events.sort(key=lambda x: x.time, reverse=False)
    return events


if __name__ == '__main__':
    config_path = f'scenarios/baseline/data/config.json'
    with open(config_path) as f:
        config = json.load(f)
        
    datetime_str = dt.datetime.strftime(dt.date.today(), '%Y%m%d')
    # sys.stdout = open(f'console_logs_{datetime_str}.log', 'w')
    
    # spd.FileLogger(name='test', filename=f'logs/REAL_{datetime_str}.log', truncate=True)
    spd.FileLogger(name='test', filename=f'logs/{datetime_str}_{config["mcts_log_name"]}', truncate=True)
    logger = spd.get('test')
    logger.set_pattern("[%l] %v")
    # logger = None

    # spd.FileLogger(name='MCTS', filename=f'logs/MCTS_{datetime_str}.log', truncate=True)
    # mcts_logger = spd.get('MCTS')
    # mcts_logger.set_pattern("[%l] %v")
    mcts_logger = None

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log_level', type=str, default='DEBUG')
    args = parser.parse_args()
    args = namespace_to_dict(args)
    if args["log_level"] == 'INFO':
        logger.set_level(spd.LogLevel.INFO)
    elif args["log_level"] == 'DEBUG':
        logger.set_level(spd.LogLevel.DEBUG)
    elif args["log_level"] == 'ERROR':
        logger.set_level(spd.LogLevel.ERR)
        
    vehicle_count = config["vehicle_count"]
    starting_date_str = config['starting_date_str']
    config_path = f'scenarios/baseline/data/trip_plan_{starting_date_str}.json'
    with open(config_path) as f:
        trip_plan = json.load(f)

    if vehicle_count != "":
        config_path = f'scenarios/baseline/data/vehicle_plan_{starting_date_str}_{vehicle_count}.json'
    else:
        config_path = f'scenarios/baseline/data/vehicle_plan_{starting_date_str}.json'
    
    with open(config_path) as f:
        bus_plan = json.load(f)

    travel_model = EmpiricalTravelModelLookup(starting_date_str, logger=None)
    # sim_environment = EnvironmentModel(travel_model, logger)
    sim_environment = EnvironmentModelFast(travel_model, logger)

    # TODO: Switch dispatch policies with NearestDispatch
    dispatch_policy = SendNearestDispatchPolicy(travel_model) # RandomDispatch(travel_model)

    # TODO: Move to environment model once i know it works
    valid_actions = None

    # event_file = config["event_file"]
    
    starting_date = dt.datetime.strptime(starting_date_str, '%Y%m%d')
    starting_time = dt.time(0, 0, 0)
    starting_datetime = dt.datetime.combine(starting_date, starting_time)

    Buses, Stops = load_initial_state(starting_date_str, bus_plan, trip_plan)

    bus_arrival_events = load_events(travel_model, starting_date_str, Buses, Stops, trip_plan)

    # HACK:
    # Injecting incident
    bus_arrival_events = manually_insert_disruption(bus_arrival_events,
                                                 buses=Buses,
                                                 bus_id='140',
                                                 time=str_timestamp_to_datetime('2021-10-18 05:45:00'))
    bus_arrival_events.sort(key=lambda x: x.time, reverse=False)
    
    # Removing arrive events and changing it to a datastruct to pass to the system
    with open(f'scenarios/baseline/data/sampled_ons_offs_dict_{starting_date_str}.pkl', 'rb') as handle:
        passenger_arrival_distribution = pickle.load(handle)
    
    # END HACK

    starting_state = copy.deepcopy(State(stops=Stops, 
                                         buses=Buses, 
                                         bus_events=bus_arrival_events, 
                                         time=bus_arrival_events[0].time))

    mcts_discount_factor = config["mcts_discount_factor"]
    # mcts_discount_factor = 1
    rollout_policy = BareMinimumRollout()
    lookahead_horizon_delta_t = 60 * 60 * 1  # 60*60*N for N hour horizon
    # lookahead_horizon_delta_t = None  # Runs until the end
    uct_tradeoff = config["uct_tradeoff"]
    pool_thread_count = 0
    iter_limit = config["iter_limit"]
    allowed_computation_time = config["allowed_computation_time"]
    mcts_type = MCTSType.MODULAR_MCTS
    mdp_environment_model = DecisionEnvironmentDynamics(travel_model,
                                                        dispatch_policy,
                                                        logger=None)
    
    decision_maker = DecisionMaker(environment_model=sim_environment,
                                   travel_model=travel_model,
                                   dispatch_policy=None,
                                   logger=None,
                                   pool_thread_count=pool_thread_count,
                                   mcts_type=mcts_type,
                                   discount_factor=mcts_discount_factor,
                                   mdp_environment_model=mdp_environment_model,
                                   rollout_policy=rollout_policy,
                                   uct_tradeoff=uct_tradeoff,
                                   iter_limit=iter_limit,
                                   lookahead_horizon_delta_t=lookahead_horizon_delta_t,
                                   allowed_computation_time=allowed_computation_time,  # 5 seconds per thread
                                   starting_date=starting_date_str,
                                   )

    simulator = Simulator(starting_event_queue=copy.deepcopy(bus_arrival_events),
                          starting_state=starting_state,
                          environment_model=sim_environment,
                          event_processing_callback=decision_maker.event_processing_callback_funct,
                          passenger_arrival_distribution=passenger_arrival_distribution,
                          valid_actions=valid_actions,
                          logger=logger,
                          minute_interval=config['minute_interval'])

    simulator.run_simulation()

    # sys.stdout.close()
