# All dates and times should just be datetime!
# from Environment.EmpiricalTravelModel import EmpiricalTravelModel
from Environment.EmpiricalTravelModelLookup import EmpiricalTravelModelLookup
from Environment.enums import BusStatus, BusType, EventType
from Environment.DataStructures.Event import Event
from Environment.DataStructures.Bus import Bus
from Environment.DataStructures.Stop import Stop
from Environment.DataStructures.State import State
from decision_making.coordinator.RandomCoord import RandomCoord
from decision_making.dispatch.SendNearestDispatchPolicy import SendNearestDispatchPolicy
from decision_making.dispatch.RandomDispatch import RandomDispatch
from decision_making.dispatch.DoNothing import DoNothing
from decision_making.ValidActions import ValidActions
from Environment.Simulator import Simulator
from Environment.EnvironmentModel import EnvironmentModel
from src.utils import *

from tqdm import tqdm
import spdlog as spd
import numpy as np
import pandas as pd
import argparse
import json
import copy
import pickle
import os


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
        bus_capacity       = bus_info['vehicle_capacity']
        bus_starting_depot = bus_info['starting_depot']
        bus_block_trips    = np.asarray(bus_info['trips'])
        
        bus_block_trips = [tuple(l) for l in bus_block_trips]
        for i, bus_block_trip in enumerate(bus_block_trips):
            block_id  = bus_block_trip[0]
            trip_id   = bus_block_trip[1]
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


def load_events(starting_date, Buses, Stops, trip_plan, random_seed=100):
    print("Adding events...")
    np.random.seed(random_seed)
    has_broken = False
    is_weekend = 0 if dt.datetime.strptime(starting_date, '%Y-%m-%d').weekday() < 5 else 1
    # Load distributions
    # sampled_loads = pd.read_pickle('scenarios/baseline/data/sampled_loads.pkl')
    sampled_loads = pd.read_pickle('scenarios/baseline/data/sampled_ons_offs.pkl')
    
    # Initial events
    # Includes: Trip starts, passenger sampling
    # all active stops that buses will pass
    events = []
    
    stop_list = []
    
    # event_file = 'events_all_vehicles.pkl'
    event_file = 'events_1.pkl'
    saved_events = f'scenarios/baseline/data/{event_file}'
    
    pbar = tqdm(Buses.items())
    if not os.path.exists(saved_events):
    # if True:
        for bus_id, bus in pbar:
            if bus.type == BusType.OVERLOAD:
                continue
            blocks_trips = bus.bus_block_trips

            # Start trip (assuming trips are in sequential order)
            block = blocks_trips[0][0]
            trip = blocks_trips[0][1]
            st   = trip_plan[trip]['scheduled_time']
            st = [str_timestamp_to_datetime(st).time().strftime('%H:%M:%S') for st in st][0]
            event_datetime = str_timestamp_to_datetime(f"{starting_date_str} {st}")
            event = Event(event_type=EventType.VEHICLE_START_TRIP, time=event_datetime)
            events.append(event)
            
            # Populate stops
            for block_trip in blocks_trips:
                block = int(block_trip[0])
                trip = block_trip[1]
                route_id         = trip_plan[trip]['route_id']
                route_direction  = trip_plan[trip]['route_direction']
                route_id_dir     = f"{route_id}_{route_direction}"
                scheduled_time   = trip_plan[trip]['scheduled_time']
                stop_id_original = trip_plan[trip]['stop_id_original']
                scheduled_time = [str_timestamp_to_datetime(st).strftime('%Y-%m-%d %H:%M:%S') for st in scheduled_time]
                
                for i in range(len(scheduled_time)):
                    load = sampled_loads.query("route_id_dir == @route_id_dir and block_abbr == @block and stop_id_original == @stop_id_original[@i] and scheduled_time == @scheduled_time[@i]").iloc[0]['sampled_loads']
                    ons = sampled_loads.query("route_id_dir == @route_id_dir and block_abbr == @block and stop_id_original == @stop_id_original[@i] and scheduled_time == @scheduled_time[@i]").iloc[0]['ons']
                    offs = sampled_loads.query("route_id_dir == @route_id_dir and block_abbr == @block and stop_id_original == @stop_id_original[@i] and scheduled_time == @scheduled_time[@i]").iloc[0]['offs']
                    
                    print(f"{block}, {stop_id_original[i]}, {scheduled_time[i]}, {route_id_dir}, {load}, {ons}, {offs}")
                    
                    pbar.set_description(f"Processing {block}, {stop_id_original[i]}, {scheduled_time[i]}, {route_id_dir}, {load}, {ons}, {offs}")
                    # making sure passengers arrives before the bus
                    event_datetime = str_timestamp_to_datetime(f"{scheduled_time[i]}") - dt.timedelta(minutes=EARLY_PASSENGER_DELTA_MIN)
                    
                    event = Event(event_type=EventType.PASSENGER_ARRIVE_STOP, 
                                time=event_datetime, 
                                type_specific_information={'route_id_dir': route_id_dir,
                                                           'stop_id': stop_id_original[i], 
                                                           'load':load, 'ons':ons, 'offs':offs})
                    events.append(event)
                    
                    # people will leave after N minutes.
                    event = Event(event_type=EventType.PASSENGER_LEAVE_STOP, 
                                time=event_datetime + dt.timedelta(minutes=PASSENGER_TIME_TO_LEAVE),
                                type_specific_information={'route_id_dir': route_id_dir,
                                                           'stop_id': stop_id_original[i], 
                                                           'time':event_datetime})
                    events.append(event)
                    
                    # # probability that a bus breaks down
                    # if np.random.uniform(0, 1) > 0.70 and not has_broken:
                    #     has_broken = True
                    #     print("BROKEN!")
                    #     event = Event(event_type=EventType.VEHICLE_BREAKDOWN, 
                    #                   time=event_datetime + dt.timedelta(minutes=np.random.randint(10, 20)),
                    #                   type_specific_information={'bus_id': bus_id})
                    #     events.append(event)
        
        events.sort(key=lambda x: x.time, reverse=False)
        # [print(event) for event in events]
        
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

    if time < start_time or time > end_time:
        raise "Time beyond limits."

    event = Event(event_type=EventType.VEHICLE_BREAKDOWN,
                  time=time,
                  type_specific_information={'bus_id': bus_id})
    events.append(event)
    events.sort(key=lambda x: x.time, reverse=False)
    return events


if __name__ == '__main__':
    datetime_str = dt.datetime.strftime(dt.date.today(), DATETIME_FORMAT)
    spd.FileLogger(name='test', filename=f'logs/BASE_{datetime_str}.log', truncate=True)
    logger = spd.get('test')
    logger.set_pattern("[%l] %v")

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log_level', type=str, default='DEBUG')
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
        
    travel_model = EmpiricalTravelModelLookup(logger)
    sim_environment = EnvironmentModel(travel_model, logger)
    
    # dispatcher  = RandomDispatch(travel_model)
    # dispatcher  = DoNothing(travel_model)
    dispatcher  = SendNearestDispatchPolicy(travel_model)
    coordinator = RandomCoord(sim_environment, travel_model, dispatcher, logger)
    
    # TODO: Move to environment model once i know it works
    valid_actions = ValidActions(travel_model, logger)
    
    starting_date_str = '2021-08-23'
    starting_date = dt.datetime.strptime(starting_date_str, '%Y-%m-%d')
    starting_time = dt.time(0, 0, 0)
    starting_datetime = dt.datetime.combine(starting_date, starting_time)
    
    Buses, Stops = load_initial_state(bus_plan, trip_plan)
    
    passenger_events = load_events(starting_date_str, Buses, Stops, trip_plan)

    # HACK:
    passenger_events = manually_insert_disruption(passenger_events,
                                                  buses=Buses,
                                                  bus_id='129',
                                                  # time=str_timestamp_to_datetime('2021-08-23 16:15:00'))
                                                  time=str_timestamp_to_datetime('2021-08-23 14:19:00'))

    starting_state = copy.deepcopy(State(Stops, Buses, events=passenger_events, time=starting_datetime))
    simulator = Simulator(starting_event_queue=copy.deepcopy(passenger_events),
                          starting_state=starting_state,
                          environment_model=sim_environment,
                          event_processing_callback=coordinator.event_processing_callback_funct,
                          valid_actions=valid_actions,
                          logger=logger)

    simulator.run_simulation()
