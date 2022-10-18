import copy
from src.utils import *
from Environment.enums import LogType, EventType


class StopDynamics:
    
    def __init__(self, travel_model, logger) -> None:
        self.travel_model = travel_model
        self.logger = logger
    
    def update_stop(self, curr_event, _new_time, full_state):
        if curr_event.event_type == EventType.PASSENGER_ARRIVE_STOP:
            additional_info   = curr_event.type_specific_information
            curr_route_id_dir = additional_info['route_id_dir']
            curr_stop_id      = additional_info['stop_id']
            curr_stop_load    = additional_info['load']
            curr_stop_ons     = additional_info['ons']
            curr_stop_offs    = additional_info['offs']
            
            passenger_waiting = full_state.stops[curr_stop_id].passenger_waiting
            if passenger_waiting == None:
                passenger_waiting = {}
                
            # Initial values for the passenger dictionary
            passenger_waiting[curr_route_id_dir] = {}
            passenger_waiting[curr_route_id_dir][_new_time] = {'got_on_bus':0, 'remaining':0, 'block_trip': "", 'ons':curr_stop_ons, 'offs':curr_stop_offs}  # 'load':curr_stop_load, 
            
            full_state.stops[curr_stop_id].passenger_waiting = passenger_waiting
            
            # log(self.logger, _new_time, f"{curr_stop_load} load for {curr_route_id_dir} at stop: {curr_stop_id}", LogType.INFO)

            return []
        
        if curr_event.event_type == EventType.PASSENGER_LEAVE_STOP:
            additional_info   = curr_event.type_specific_information
            curr_route_id_dir = additional_info['route_id_dir']
            curr_stop_id      = additional_info['stop_id']
            time_key          = additional_info['time']
            
            # left_behind = additional_info['left_behind']
            passenger_waiting = full_state.stops[curr_stop_id].passenger_waiting

            # HACK: Not sure if this is correct
            if passenger_waiting is None:
                return []

            if time_key in passenger_waiting[curr_route_id_dir]:
                remaining = passenger_waiting[curr_route_id_dir][time_key]['remaining']
                got_on_bus = passenger_waiting[curr_route_id_dir][time_key]['got_on_bus']
                ons = passenger_waiting[curr_route_id_dir][time_key]['ons']
                
                if got_on_bus == 0:
                    remaining = ons
                
                # Count remaining people as walk-offs
                full_state.stops[curr_stop_id].total_passenger_walk_away += remaining
                
                # Delete dictionary for this time
                del full_state.stops[curr_stop_id].passenger_waiting[curr_route_id_dir][time_key]

                if remaining > 0:
                    log(self.logger, _new_time, f"Stop @ {curr_stop_id} for {curr_route_id_dir}: {remaining} people left after {PASSENGER_TIME_TO_LEAVE} minutes.", LogType.ERROR)
                    
            return []