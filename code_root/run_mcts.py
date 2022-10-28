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


# TODO: Have the ability to save and load states from file.
def load_initial_state(bus_plan, trip_plan, random_seed=100):
    print("Loading initial states...")
    active_stops = []

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
                t_state_change = str_timestamp_to_datetime(f"{starting_date_str} {st}")

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


def load_events(event_file, starting_date, Buses, Stops, trip_plan, random_seed=100):
    print("Adding events...")
    np.random.seed(random_seed)
    has_broken = False
    is_weekend = 0 if dt.datetime.strptime(starting_date, '%Y-%m-%d').weekday() < 5 else 1
    # Load distributions
    with open('scenarios/baseline/data/sampled_ons_offs_dict.pkl', 'rb') as handle:
        sampled_travel_time = pickle.load(handle)

    # Initial events
    # Includes: Trip starts, passenger sampling
    # all active stops that buses will pass
    events = []
    saved_events = f'scenarios/baseline/data/{event_file}'

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
            event_datetime = str_timestamp_to_datetime(f"{starting_date_str} {st}")
            event = Event(event_type=EventType.VEHICLE_START_TRIP,
                          time=event_datetime,
                          type_specific_information={'bus_id': bus_id})
            events.append(event)

            # # Populate stops
            # for block_trip in blocks_trips:
            #     block = int(block_trip[0])
            #     trip = block_trip[1]
            #     route_id = trip_plan[trip]['route_id']
            #     route_direction = trip_plan[trip]['route_direction']
            #     route_id_dir = f"{route_id}_{route_direction}"
            #     scheduled_time = trip_plan[trip]['scheduled_time']
            #     stop_id_original = trip_plan[trip]['stop_id_original']
            #     scheduled_time = [str_timestamp_to_datetime(st).strftime('%Y-%m-%d %H:%M:%S') for st in scheduled_time]

            #     for stop_sequence in range(len(scheduled_time)):
            #         # sampled_travel_time['23_FROM DOWNTOWN', 2310, 32, 'DWMRT', pd.Timestamp('2021-08-23 05:41:00')]
            #         val = sampled_travel_time[route_id_dir,
            #                                   block,
            #                                   stop_sequence + 1,
            #                                   stop_id_original[stop_sequence],
            #                                   pd.Timestamp(scheduled_time[stop_sequence])]
            #         load = val['sampled_loads']
            #         ons = val['ons']
            #         offs = val['offs']
            #         print(f"{block}, {stop_id_original[stop_sequence]}, {scheduled_time[stop_sequence]}, {route_id_dir}, {load}, {ons}, {offs}")

            #         pbar.set_description(
            #             f"Processing {block}, {stop_id_original[stop_sequence]}, {scheduled_time[stop_sequence]}, {route_id_dir}, {load}, {ons}, {offs}")
            #         # making sure passengers arrives before the bus
            #         event_datetime = str_timestamp_to_datetime(f"{scheduled_time[stop_sequence]}") - dt.timedelta(
            #             minutes=EARLY_PASSENGER_DELTA_MIN)

            #         event = Event(event_type=EventType.PASSENGER_ARRIVE_STOP,
            #                       time=event_datetime,
            #                       type_specific_information={'route_id_dir': route_id_dir,
            #                                                  'block_abbr': block,
            #                                                  'stop_sequence': stop_sequence + 1,
            #                                                  'stop_id': stop_id_original[stop_sequence],
            #                                                  'load': load, 'ons': ons, 'offs': offs})
            #         events.append(event)

        events.sort(key=lambda x: x.time, reverse=False)

        with open(saved_events, "wb") as f:
            pickle.dump(events, f)
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
    datetime_str = dt.datetime.strftime(dt.date.today(), DATETIME_FORMAT)
    # sys.stdout = open(f'console_logs_{datetime_str}.log', 'w')
    
    spd.FileLogger(name='test', filename=f'logs/REAL_{datetime_str}.log', truncate=True)
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

    config_name = "config_2.json"
    config_path = f'scenarios/baseline/data/{config_name}'
    with open(config_path) as f:
        config = json.load(f)

    config_path = f'scenarios/baseline/data/{config["trip_plan"]}'
    with open(config_path) as f:
        trip_plan = json.load(f)

    config_path = f'scenarios/baseline/data/{config["vehicle_plan"]}'
    with open(config_path) as f:
        bus_plan = json.load(f)

    travel_model = EmpiricalTravelModelLookup(config_name, logger=None)
    # sim_environment = EnvironmentModel(travel_model, logger)
    sim_environment = EnvironmentModelFast(travel_model, logger)

    # TODO: Switch dispatch policies with NearestDispatch
    dispatch_policy = RandomDispatch(travel_model)

    # TODO: Move to environment model once i know it works
    valid_actions = None

    event_file = config["event_file"]
    
    starting_date_str = config["schedule_date"]
    starting_date = dt.datetime.strptime(starting_date_str, '%Y-%m-%d')
    starting_time = dt.time(0, 0, 0)
    starting_datetime = dt.datetime.combine(starting_date, starting_time)

    Buses, Stops = load_initial_state(bus_plan, trip_plan)

    bus_arrival_events = load_events(event_file, starting_date_str, Buses, Stops, trip_plan)

    # HACK:
    # Removing leave events
    # passenger_events = [pe for pe in passenger_events if pe.event_type != EventType.PASSENGER_LEAVE_STOP]
    # Injecting incident
    bus_arrival_events = manually_insert_disruption(bus_arrival_events,
                                                  buses=Buses,
                                                  bus_id='129',
                                                  time=str_timestamp_to_datetime('2021-08-23 14:20:00'))
    # Add one last event to ensure everyone leaves
    # event = Event(event_type=EventType.PASSENGER_LEAVE_STOP,
    #               time=passenger_events[-1].time + dt.timedelta(minutes=PASSENGER_TIME_TO_LEAVE))
    # bus_arrival_events.append(event)
    bus_arrival_events.sort(key=lambda x: x.time, reverse=False)
    
    # Removing arrive events and changing it to a datastruct to pass to the system
    with open('scenarios/baseline/data/sampled_ons_offs_dict.pkl', 'rb') as handle:
        passenger_arrival_distribution = pickle.load(handle)
    
    # END HACK

    starting_state = copy.deepcopy(State(stops=Stops, 
                                         buses=Buses, 
                                         bus_events=bus_arrival_events, 
                                         time=bus_arrival_events[0].time))

    mcts_discount_factor = 0.99997
    # mcts_discount_factor = 1
    rollout_policy = BareMinimumRollout()
    lookahead_horizon_delta_t = 60 * 60 * 1  # 60*60*N for N hour horizon
    # lookahead_horizon_delta_t = None  # Runs until the end
    uct_tradeoff = 1.44
    pool_thread_count = 0
    iter_limit = 200
    allowed_computation_time = 15
    mcts_type = MCTSType.MODULAR_MCTS
    mdp_environment_model = DecisionEnvironmentDynamics(travel_model,
                                                        dispatch_policy,
                                                        logger=None)
    
    event_chain_dir = config["event_chain_dir"]
    
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
                                   allowed_computation_time=allowed_computation_time,  # 5 seconds per thread,
                                   event_chain_dir=event_chain_dir
                                   )

    simulator = Simulator(starting_event_queue=copy.deepcopy(bus_arrival_events),
                          starting_state=starting_state,
                          environment_model=sim_environment,
                          event_processing_callback=decision_maker.event_processing_callback_funct,
                          passenger_arrival_distribution=passenger_arrival_distribution,
                          valid_actions=valid_actions,
                          logger=logger)

    simulator.run_simulation()

    # sys.stdout.close()