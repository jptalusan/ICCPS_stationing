import time
from Environment.enums import BusStatus
from src.utils import *
import datetime as dt
import spdlog as spd

class Simulator:
    
    def __init__(self,
                 starting_state,
                 environment_model,
                 event_processing_callback,
                 starting_event_queue,
                 logger) -> None:
        
        self.state = starting_state
        self.environment_model = environment_model
        self.event_processing_callback = event_processing_callback
        self.event_queue = starting_event_queue
        self.logger = logger
        
        self.start_sim_time = None
        self.starting_num_events = len(starting_event_queue)
        self.num_events_processed = 0
        
        spd.FileLogger(name='visualizer', filename='visualizer.csv', truncate=True)
        self.visual_log = spd.get('visualizer')
        self.visual_log.set_pattern("%v")
        self.visual_log.set_level(spd.LogLevel.DEBUG)
        self.visual_log.debug(f"time,id,trip_id,last_visited_stop,value,fraction,icon")
        
    # Other actors catch up to the event..?
    def run_simulation(self):
        log(self.logger, dt.datetime.now(), "Running simulation (real world time)")
        
        self.start_sim_time = time.time()
        
        while len(self.event_queue) > 0:
            self.update_sim_info()
        
            curr_event = self.event_queue.pop(0)
            
            new_events = self.environment_model.update(self.state, curr_event)
            
            for event in new_events:
                self.add_event(event)
            
            new_events = self.event_processing_callback(self.state, curr_event)
            
            for event in new_events:
                self.add_event(event)

            self.save_visualization(curr_event.time)
        self.print_states()
        log(self.logger, dt.datetime.now(), "Finished simulation (real world time)")
        
    def update_sim_info(self):
        self.num_events_processed += 1
        
    def add_event(self, new_event):
        self.event_queue.append(new_event)
        self.event_queue.sort(key=lambda _: _.time, reverse=False)
        
    def print_states(self):
        for bus_id, bus_obj in self.state.buses.items():
            log(self.logger, dt.datetime.now(), f"--Bus ID: {bus_id}--")
            log(self.logger, dt.datetime.now(), f"dwell_time: {bus_obj.dwell_time:.2f}")
            log(self.logger, dt.datetime.now(), f"delay_time: {bus_obj.delay_time:.2f}")
            log(self.logger, dt.datetime.now(), f"total_service_time: {bus_obj.total_service_time:.2f}")
            log(self.logger, dt.datetime.now(), f"total_passengers_served: {bus_obj.total_passengers_served}")
            log(self.logger, dt.datetime.now(), f"block_trips: {bus_obj.bus_block_trips}")

        # for stop_id, stop_obj in self.state.stops.items():
        #     log(self.logger, dt.datetime.now(), f"--Stop ID: {stop_id}--")
        #     log(self.logger, dt.datetime.now(), f"total_passenger_ons: {stop_obj.total_passenger_ons}")
        #     log(self.logger, dt.datetime.now(), f"total_passenger_offs: {stop_obj.total_passenger_offs}")
        #     log(self.logger, dt.datetime.now(), f"total_passenger_walk_away: {stop_obj.total_passenger_walk_away}")
            
    def save_visualization(self, event_time):
        # self.visual_log.debug(f"time,id,trip_id,last_visited_stop,value,fraction,icon")
        for bus_id, bus_obj in self.state.buses.items():
            if bus_obj.status != BusStatus.IDLE:
                current_trip = ""
                if bus_obj.current_block_trip:
                    current_trip = bus_obj.current_block_trip[1]
                self.visual_log.debug(f"{event_time},{bus_id},{current_trip},{bus_obj.current_stop},{bus_obj.current_load},{bus_obj.percent_to_next_stop},car")
            
        for stop_id, stop_obj in self.state.stops.items():
            pw = stop_obj.passenger_waiting
            ons = 0
            if pw:
                for k, v in pw.items():
                    if len(v) > 0:
                        for j, w in pw[k].items():
                            ons = w['ons']
                            self.visual_log.debug(f"{event_time},{stop_id},,{stop_id},{ons},,add-person")