import time
from Environment.enums import BusStatus, BusType
from src.utils import *
import datetime as dt
import spdlog as spd

class Simulator:
    
    def __init__(self,
                 starting_state,
                 environment_model,
                 event_processing_callback,
                 starting_event_queue,
                 valid_actions,
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
        self.visual_log.debug(f"time,id,trip_id,last_visited_stop,value,fraction,icon,radius")
        
        self.last_visual_log = None
        self.valid_actions = valid_actions
        
    # Other actors catch up to the event..?
    def run_simulation(self):
        log(self.logger, dt.datetime.now(), "Running simulation (real world time)")
        
        self.start_sim_time = time.time()
        
        # initialize state
        while len(self.event_queue) > 0:
            self.update_sim_info()

            valid_actions = self.valid_actions.get_valid_actions(self.state)
            
            new_events = self.event_processing_callback(valid_actions, self.state)
            
            for event in new_events:
                self.add_event(event)
            
            update_event = self.event_queue.pop(0)
            reward, new_events = self.environment_model.update(self.state, update_event)
            
            for event in new_events:
                self.add_event(event)

            self.save_visualization(update_event.time, granularity_s=None)
            
        self.print_states()
        log(self.logger, dt.datetime.now(), "Finished simulation (real world time)")
        
    def update_sim_info(self):
        self.num_events_processed += 1
        
    def add_event(self, new_event):
        self.event_queue.append(new_event)
        self.event_queue.sort(key=lambda _: _.time, reverse=False)
        
    def print_states(self):
        LOGTYPE = LogType.INFO
        log(self.logger, dt.datetime.now(), f"Total events processed: {self.num_events_processed}", LOGTYPE)
        for bus_id, bus_obj in self.state.buses.items():
            log(self.logger, dt.datetime.now(), f"--Bus ID: {bus_id}--", LOGTYPE)
            log(self.logger, dt.datetime.now(), f"total dwell_time: {bus_obj.dwell_time:.2f} s", LOGTYPE)
            log(self.logger, dt.datetime.now(), f"aggregate delay_time: {(bus_obj.delay_time/bus_obj.total_stops):.2f} s", LOGTYPE)
            log(self.logger, dt.datetime.now(), f"total_service_time: {bus_obj.total_service_time:.2f}", LOGTYPE)
            log(self.logger, dt.datetime.now(), f"total_passengers_served: {bus_obj.total_passengers_served}", LOGTYPE)
            log(self.logger, dt.datetime.now(), f"total_servicekms_moved: {bus_obj.total_servicekms_moved:.2f} km", LOGTYPE)
            log(self.logger, dt.datetime.now(), f"total_deadkms_moved: {bus_obj.total_deadkms_moved:.2f} km", LOGTYPE)

        # for stop_id, stop_obj in self.state.stops.items():
        #     log(self.logger, dt.datetime.now(), f"--Stop ID: {stop_id}--", LOGTYPE)
        #     log(self.logger, dt.datetime.now(), f"total_passenger_ons: {stop_obj.total_passenger_ons}", LOGTYPE)
        #     log(self.logger, dt.datetime.now(), f"total_passenger_offs: {stop_obj.total_passenger_offs}", LOGTYPE)
        #     log(self.logger, dt.datetime.now(), f"total_passenger_walk_away: {stop_obj.total_passenger_walk_away}", LOGTYPE)
            
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