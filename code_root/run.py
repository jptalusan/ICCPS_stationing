
# All dates and times should just be datetime!
from random import sample
from Environment.EmpiricalTravelModel import EmpiricalTravelModel
from Environment.enums import BusStatus, BusType, EventType
from Environment.DataStructures.Event import Event
from Environment.DataStructures.Bus import Bus
from Environment.DataStructures.Stop import Stop
from Environment.DataStructures.State import State
from decision_making.coordinator.DispatchOnlyCoord import DispatchOnlyCoord
from decision_making.dispatch.SendNearestDispatchPolicy import SendNearestDispatchPolicy
from decision_making.coordinator.DoNothing import DoNothing as Coord_DoNothing
from decision_making.dispatch.DoNothing import DoNothing as Dispatch_DoNothing
from Environment.Simulator import Simulator
from Environment.EnvironmentModel import EnvironmentModel
import argparse
from src.utils import *
import json
import copy
import spdlog as spd
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
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
        bus_capacity = bus_info['vehicle_capacity']
        bus_starting_depot = bus_info['starting_depot']
        bus_block_trips = np.asarray(bus_info['trips'])
        
        bus_block_trips = [tuple(l) for l in bus_block_trips]
        for i, bus_block_trip in enumerate(bus_block_trips):
            block_id = bus_block_trip[0]
            trip_id = bus_block_trip[1]
            trip_info = trip_plan[trip_id]
            stop_id_original = trip_info['stop_id_original']
            active_stops.extend(stop_id_original)
            if i == 0:
                st   = trip_plan[trip_id]['scheduled_time']
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
    sampled_offs = pd.read_pickle('scenarios/baseline/data/sampled_offs.pkl')
    sampled_ons  = pd.read_pickle('scenarios/baseline/data/sampled_ons.pkl')
    sampled_offs['time'] = pd.to_datetime(sampled_offs['time'], format='%H:%M:%S')
    sampled_ons['time']  = pd.to_datetime(sampled_ons['time'], format='%H:%M:%S')
    
    # Initial events
    # Includes: Trip starts, passenger sampling
    # all active stops that buses will pass
    events = []
    
    stop_list = []
    
    # event_file = 'events_all_vehicles.pkl'
    event_file = 'events_2.pkl'
    saved_events = f'/media/seconddrive/JP/gits/mta_simulator_redo/code_root/scenarios/baseline/data/{event_file}'
    
    if not os.path.exists(saved_events):
        for bus_id, bus in tqdm(Buses.items()):
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
                scheduled_time = [str_timestamp_to_datetime(st).time().strftime('%H:%M:%S') for st in scheduled_time]
                
                for i in range(len(scheduled_time)):
                    offs_df = sampled_offs.query("route_id_direction == @route_id_dir and block_abbr == @block and stop_id_original == @stop_id_original[@i] and IsWeekend == @is_weekend")
                    offs_df = offs_df[offs_df['time'] == f'1900-01-01 {scheduled_time[i]}']
                    offs = offs_df.iloc[0]['sample_offs']
                    
                    ons_df = sampled_ons.query("route_id_direction == @route_id_dir and block_abbr == @block and stop_id_original == @stop_id_original[@i] and IsWeekend == @is_weekend")
                    ons  = ons_df.iloc[0]['sample_ons']
                    # print(f"{scheduled_time[i]}, {ons}, {load}")
                    
                    # making sure passengers arrives before the bus
                    event_datetime = str_timestamp_to_datetime(f"{starting_date_str} {scheduled_time[i]}") - dt.timedelta(minutes=EARLY_PASSENGER_DELTA_MIN)
                    
                    event = Event(event_type=EventType.PASSENGER_ARRIVE_STOP, 
                                time=event_datetime, 
                                type_specific_information={'route_id_dir': route_id_dir,
                                                           'stop_id': stop_id_original[i], 
                                                           'ons':ons, 'offs':offs})
                    events.append(event)
                    
                    # people will leave after N minutes.
                    event = Event(event_type=EventType.PASSENGER_LEAVE_STOP, 
                                time=event_datetime + dt.timedelta(minutes=PASSENGER_TIME_TO_LEAVE),
                                type_specific_information={'route_id_dir': route_id_dir,
                                                           'stop_id': stop_id_original[i], 
                                                           'time':event_datetime})
                    events.append(event)
                    
                    # probability that a bus breaks down
                    if np.random.uniform(0, 1) > 0.70 and not has_broken:
                        has_broken = True
                        print("BROKEN!")
                        event = Event(event_type=EventType.VEHICLE_BREAKDOWN, 
                                      time=event_datetime + dt.timedelta(minutes=np.random.randint(30, 60)),
                                      type_specific_information={'bus_id': bus_id})
                        events.append(event)
        
        events.sort(key=lambda x: x.time, reverse=False)
        # [print(event) for event in events]
        
        with open(saved_events, "wb") as f:
            pickle.dump(events, f)
    else:
        print("loading events...")
        with open(saved_events, "rb") as f:
            events = pickle.load(f)
            
    return events

if __name__ == '__main__':
    spd.FileLogger(name='test', filename='spdlog_example.log', truncate=True)
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
        
    travel_model = EmpiricalTravelModel(logger)
    sim_environment = EnvironmentModel(travel_model, logger)
    # dispatcher = Dispatch_DoNothing(travel_model)
    # coordinator = Coord_DoNothing(sim_environment, travel_model, dispatcher, logger)
    
    dispatcher = SendNearestDispatchPolicy(travel_model)
    coordinator = DispatchOnlyCoord(sim_environment, travel_model, dispatcher, logger)
    
    starting_date_str = '2021-08-23'
    starting_date = dt.datetime.strptime(starting_date_str, '%Y-%m-%d')
    starting_time = dt.time(0, 0, 0)
    starting_datetime = dt.datetime.combine(starting_date, starting_time)
    
    Buses, Stops = load_initial_state(bus_plan, trip_plan)
    
    incident_events = load_events(starting_date_str, Buses, Stops, trip_plan)
    
    starting_state = copy.deepcopy(State(Stops, Buses, events=None, time=starting_datetime))
    simulator = Simulator(starting_event_queue=copy.deepcopy(incident_events),
                          starting_state=starting_state,
                          environment_model=sim_environment,
                          event_processing_callback=coordinator.event_processing_callback_funct,
                          logger=logger)

    simulator.run_simulation()