import copy
import time

import pandas as pd
from Environment.enums import BusStatus, BusType, ActionType, EventType
from src.utils import *
import datetime as dt
import spdlog as spd
import numpy as np

class Simulator:
    
    def __init__(self,
                 starting_state,
                 environment_model,
                 event_processing_callback,
                 starting_event_queue,
                 passenger_arrival_distribution,
                 valid_actions,
                 logger,
                 minute_interval,
                 log_name) -> None:
        
        self.state = starting_state
        self.environment_model = environment_model
        self.event_processing_callback = event_processing_callback
        self.event_queue = starting_event_queue
        self.passenger_arrival_distribution = passenger_arrival_distribution
        self.logger = logger
        
        self.start_sim_time = None
        self.starting_num_events = len(starting_event_queue)
        self.num_events_processed = 0
        self.num_decision_epochs = 0
        self.minute_interval = minute_interval
        self.log_name = log_name

        spd.FileLogger(name='visualizer', filename='visualizer.csv', truncate=True)
        self.visual_log = spd.get('visualizer')
        self.visual_log.set_pattern("%v")
        self.visual_log.set_level(spd.LogLevel.DEBUG)
        self.visual_log.debug(f"time,id,trip_id,last_visited_stop,value,fraction,icon,radius")

        spd.FileLogger(name='stop_metrics', filename=f'logs/stop_metrics_{self.log_name}.csv', truncate=True)
        self.stop_metrics_log = spd.get('stop_metrics')
        self.stop_metrics_log.set_pattern("%v")
        self.stop_metrics_log.set_level(spd.LogLevel.DEBUG)
        self.stop_metrics_log.debug(f"state_time,stop_id,total_ons,total_offs,total_walkaway,ons,offs,remaining")
        
        spd.FileLogger(name='bus_metrics', filename=f'logs/bus_metrics_{self.log_name}.csv', truncate=True)
        self.bus_metrics_log = spd.get('bus_metrics')
        self.bus_metrics_log.set_pattern("%v")
        self.bus_metrics_log.set_level(spd.LogLevel.DEBUG)
        self.bus_metrics_log.debug(f"state_time,bus_id,status,type,capacity,load,current_block,current_trip,current_stop,time_at_last_stop,total_passengers_served,deadkms,servicekms")

        spd.FileLogger(name='action_taken', filename=f'logs/action_taken_{self.log_name}.csv', truncate=True)
        self.action_taken_log = spd.get('action_taken')
        self.action_taken_log.set_pattern("%v")
        self.action_taken_log.set_level(spd.LogLevel.DEBUG)
        self.action_taken_log.debug(f"state_time,action")
        
        self.last_visual_log = None
        self.valid_actions = valid_actions
        
    # Other actors catch up to the event..?
    def run_simulation(self):
        log(self.logger, dt.datetime.now(), "Running simulation (real world time)", LogType.INFO)
        
        self.start_sim_time = time.time()

        # initialize state
        chosen_action = None
        update_event = None
        
        # Initialize timepoints (N minute intervals)
        timepoint_array = self.generate_timepoints(self.event_queue, minute_intervals=self.minute_interval)
        
        while len(self.event_queue) > 0:
            self.update_sim_info()

            if self.valid_actions is not None:
                _valid_actions = self.valid_actions.get_valid_actions(self.state)
            else:
                _valid_actions = None

            if update_event and (update_event.time >= timepoint_array[self.num_decision_epochs]):
                self.num_decision_epochs += 1
                chosen_action = self.event_processing_callback(_valid_actions,
                                                               self.state,
                                                               action_type=ActionType.OVERLOAD_ALLOCATE)
            elif update_event:
                chosen_action = self.event_processing_callback(_valid_actions,
                                                               self.state,
                                                               action_type=ActionType.OVERLOAD_DISPATCH)

            if chosen_action is None:
                chosen_action = {'type': ActionType.NO_ACTION, 'overload_bus': None, 'info': None}
            log(self.logger, self.state.time, f"Chosen action:{chosen_action}", LogType.DEBUG)
            new_events, _ = self.environment_model.take_action(self.state, chosen_action)
            for event in new_events:
                self.add_event(event)

            # self.action_taken_log.debug(f"{self.state.time},{chosen_action}")

            update_event = self.event_queue.pop(0)
            new_events = self.environment_model.update(self.state, update_event, self.passenger_arrival_distribution)
            for event in new_events:
                self.add_event(event)

            self.state.bus_events = copy.copy(self.event_queue)
                
            # self.save_visualization(update_event.time, granularity_s=None)
            
            # self.log_metrics()
            # print(f"Events left: {len(self.event_queue)}")
        print("Done")
            
        self.print_states()
        log(self.logger, dt.datetime.now(), "Finished simulation (real world time)", LogType.INFO)
        
    def generate_timepoints(self, event_queue, minute_intervals=15):
        start_time = event_queue[0].time
        base_time = start_time.replace(minute=0, second=0, microsecond=0)
        end_time = start_time.replace(hour=2, minute=0, second=0, microsecond=0) + dt.timedelta(days=1)
        time_array = np.arange(base_time, end_time, np.timedelta64(minute_intervals, 'm'))
        return time_array
        
    def update_sim_info(self):
        self.num_events_processed += 1
        
    def add_event(self, new_event):
        self.event_queue.append(new_event)
        self.event_queue.sort(key=lambda _: _.time, reverse=False)
        
    def log_metrics(self):
        time = self.state.time
        for stop_id, stop_obj in self.state.stops.items():
            passenger_waiting = stop_obj.passenger_waiting
            ons = 0
            offs = 0
            remaining = 0
            if passenger_waiting:
                for route_id_dir in passenger_waiting:
                    for passenger_arrival_time, sampled_data in passenger_waiting[route_id_dir].items():
                        remaining += sampled_data['remaining']
                        ons += sampled_data['ons']
                        offs += sampled_data['offs']
                
            # self.stop_metrics_log.debug(f"{time},{stop_id},{ons},{offs},{remaining}")
            self.stop_metrics_log.debug(f"{time},{stop_obj},{ons},{offs},{remaining}")
            
        for bus_id, bus_obj in self.state.buses.items():
            status = bus_obj.status
            bus_type = bus_obj.type
            current_load = bus_obj.current_load
            if bus_obj.current_block_trip:
                current_block = bus_obj.current_block_trip[0]
                current_trip = bus_obj.current_block_trip[1]
            else:
                current_block = None
                current_trip = None
                
            current_stop = bus_obj.current_stop
            time_at_last_stop = bus_obj.time_at_last_stop
            total_passengers_served = bus_obj.total_passengers_served
            capacity = bus_obj.capacity
            deadkms_moved = bus_obj.total_deadkms_moved
            servicekms_moved = bus_obj.total_servicekms_moved

            self.bus_metrics_log.debug(f"{time},{bus_id},{status},{bus_type},{capacity},{current_load},{current_block},{current_trip},{current_stop},{time_at_last_stop},{total_passengers_served},{deadkms_moved},{servicekms_moved},")

    def print_states(self):
        LOGTYPE = LogType.INFO
        log(self.logger, dt.datetime.now(), f"Total events processed: {self.num_events_processed}", LOGTYPE)
        log(self.logger, dt.datetime.now(), f"Total decision epochs: {self.num_decision_epochs}", LOGTYPE)
        for bus_id, bus_obj in self.state.buses.items():
            log(self.logger, dt.datetime.now(), f"--Bus ID: {bus_id}--", LOGTYPE)
            log(self.logger, dt.datetime.now(), f"total dwell_time: {bus_obj.dwell_time:.2f} s", LOGTYPE)
            log(self.logger, dt.datetime.now(), f"aggregate delay_time: {(bus_obj.delay_time/bus_obj.total_stops):.2f} s", LOGTYPE)
            log(self.logger, dt.datetime.now(), f"total_service_time: {bus_obj.total_service_time:.2f}", LOGTYPE)
            log(self.logger, dt.datetime.now(), f"total_passengers_served: {bus_obj.total_passengers_served}", LOGTYPE)
            log(self.logger, dt.datetime.now(), f"total_servicekms_moved: {bus_obj.total_servicekms_moved:.2f} km", LOGTYPE)
            log(self.logger, dt.datetime.now(), f"total_deadkms_moved: {bus_obj.total_deadkms_moved:.2f} km", LOGTYPE)
            log(self.logger, dt.datetime.now(), f"current_stop: {bus_obj.current_stop}", LOGTYPE)
            log(self.logger, dt.datetime.now(), f"status: {bus_obj.status}", LOGTYPE)

        for stop_id, stop_obj in self.state.stops.items():
            if stop_obj.total_passenger_walk_away > 0:
                log(self.logger, dt.datetime.now(), f"--Stop ID: {stop_id}--", LOGTYPE)
                log(self.logger, dt.datetime.now(), f"total_passenger_ons: {stop_obj.total_passenger_ons}", LOGTYPE)
                log(self.logger, dt.datetime.now(), f"total_passenger_offs: {stop_obj.total_passenger_offs}", LOGTYPE)
                log(self.logger, dt.datetime.now(), f"total_passenger_walk_away: {stop_obj.total_passenger_walk_away}", LOGTYPE)

        total_walk_aways = 0
        total_arrivals = 0
        total_boardings = 0
        for stop_id, stop_obj in self.state.stops.items():
            total_walk_aways += stop_obj.total_passenger_walk_away
            total_boardings += stop_obj.total_passenger_ons
            total_arrivals += stop_obj.total_passenger_ons + stop_obj.total_passenger_walk_away
                
        log(self.logger, dt.datetime.now(), f"Count of all passengers: {total_arrivals}", LOGTYPE)
        log(self.logger, dt.datetime.now(), f"Count of all passengers who boarded: {total_boardings}", LOGTYPE)
        log(self.logger, dt.datetime.now(), f"Count of all passengers who left: {total_walk_aways}", LOGTYPE)
            
        # for stop_id, stop_obj in self.state.stops.items():
        #     log(self.logger, dt.datetime.now(), f"--Stop ID: {stop_id}--", LOGTYPE)
        #     log(self.logger, dt.datetime.now(), f"total_passenger_ons: {stop_obj.total_passenger_ons}", LOGTYPE)
        #     log(self.logger, dt.datetime.now(), f"total_passenger_offs: {stop_obj.total_passenger_offs}", LOGTYPE)
        #     log(self.logger, dt.datetime.now(), f"total_passenger_walk_away: {stop_obj.total_passenger_walk_away}", LOGTYPE)
            # total_walk_aways += stop_obj.total_passenger_walk_away
            # total_boardings += stop_obj.total_passenger_ons
            # total_arrivals += total_boardings + total_walk_aways
        # log(self.logger, dt.datetime.now(), f"Count of all passengers: {total_arrivals}", LOGTYPE)
        # log(self.logger, dt.datetime.now(), f"Count of all passengers who boarded: {total_boardings}", LOGTYPE)
        # log(self.logger, dt.datetime.now(), f"Count of all passengers who left: {total_walk_aways}", LOGTYPE)
            
    def save_visualization(self, event_time, granularity_s=None):
        if not self.last_visual_log:
            self.last_visual_log = event_time

        if granularity_s:
            if event_time - self.last_visual_log < dt.timedelta(seconds=granularity_s):
                return 
        
        # print("Visualizer:", (event_time - self.last_visual_log))
        # self.visual_log.debug(f"time,id,trip_id,last_visited_stop,value,fraction,icon")
        for bus_id, bus_obj in self.state.buses.items():
            if bus_obj.status != BusStatus.IDLE:
                current_trip = ""
                if bus_obj.current_block_trip:
                    current_trip = bus_obj.current_block_trip[1]

                if bus_obj.type == BusType.REGULAR:
                    if bus_obj.status == BusStatus.BROKEN:
                        self.visual_log.debug(f"{event_time},{bus_id},{current_trip},{bus_obj.current_stop},100,{bus_obj.percent_to_next_stop},alert,60")
                    else:
                        self.visual_log.debug(f"{event_time},{bus_id},{current_trip},{bus_obj.current_stop},{bus_obj.current_load},{bus_obj.percent_to_next_stop},car,35")
                elif bus_obj.type == BusType.OVERLOAD:
                    self.visual_log.debug(f"{event_time},{bus_id},{current_trip},{bus_obj.current_stop},{bus_obj.current_load},{bus_obj.percent_to_next_stop},car-suv,35")
            
        for stop_id, stop_obj in self.state.stops.items():
            pw = stop_obj.passenger_waiting
            ons = 0
            if pw:
                for k, v in pw.items():
                    if len(v) > 0:
                        for j, w in pw[k].items():
                            ons = w['load']
                            remaining = w['remaining']
                            if remaining == 0:
                                self.visual_log.debug(f"{event_time},{stop_id},,{stop_id},{ons},,add-person,25")
                            else:
                                self.visual_log.debug(f"{event_time},{stop_id},,{stop_id},{remaining},,add-person,25")

        if granularity_s:
            self.last_visual_log = event_time