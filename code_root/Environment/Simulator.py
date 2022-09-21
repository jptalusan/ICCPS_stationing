import time
from src.utils import *
import datetime as dt

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
        
    # Other actors catch up to the event..?
    def run_simulation(self):
        self.logger.info("Running simulation.")
        
        self.start_sim_time = time.time()
        
        while len(self.event_queue) > 0:
            self.update_sim_info()
        
            curr_event = self.event_queue.pop(0)
                        
            # for b_id, b_obj in self.state.buses.items():
            #     self.logger.debug(f"{b_obj}")
                
            new_events = self.environment_model.update(self.state, curr_event)
            
            if new_events:
                for event in new_events:
                    self.add_event(event)
                
            # self.logger.debug("after")
            # for b_id, b_obj in self.state.buses.items():
            #     self.logger.debug(f"{b_obj}")
            # self.logger.debug(f".--.")
            
            new_events = self.event_processing_callback(self.state, curr_event)
            
            for event in new_events:
                self.add_event(event)
            pass
        
        self.print_states()
        
    def update_sim_info(self):
        self.num_events_processed += 1
        
    def add_event(self, new_event):
        self.event_queue.append(new_event)

        self.event_queue.sort(key=lambda _: _.time, reverse=False)
        
    def print_states(self):
        for bus_id, bus_obj in self.state.buses.items():
            log(self.logger, dt.datetime.now(), f"--Bus ID: {bus_id}--")
            log(self.logger, dt.datetime.now(), f"dwell_time: {bus_obj.dwell_time}")
            log(self.logger, dt.datetime.now(), f"delay_time: {bus_obj.delay_time}")
            log(self.logger, dt.datetime.now(), f"total_service_time: {bus_obj.total_service_time}")
            log(self.logger, dt.datetime.now(), f"total_passengers_served: {bus_obj.total_passengers_served}")
            
        # for stop_id, stop_obj in self.state.stops.items():
        #     log(self.logger, dt.datetime.now(), f"--Stop ID: {stop_id}--")
        #     log(self.logger, dt.datetime.now(), f"total_passenger_ons: {stop_obj.total_passenger_ons}")
        #     log(self.logger, dt.datetime.now(), f"total_passenger_offs: {stop_obj.total_passenger_offs}")
        #     log(self.logger, dt.datetime.now(), f"total_passenger_walk_away: {stop_obj.total_passenger_walk_away}")
        #     log(self.logger, dt.datetime.now(), f"passenger_waiting: {stop_obj.passenger_waiting}")