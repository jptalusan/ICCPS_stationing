import copy
from src.utils import *
from Environment.enums import LogType, EventType

class StopDynamics:
    
    def __init__(self, travel_model, logger) -> None:
        self.travel_model = travel_model
        self.logger = logger
    
    def update_stop(self, curr_event, _new_time, full_state):
        
        if curr_event.event_type == EventType.PASSENGER_ARRIVE_STOP:
            additional_info = curr_event.type_specific_information
            curr_stop_id = additional_info['stop_id']
            curr_stop_ons = additional_info['ons']
            curr_stop_load = additional_info['load']
            
            curr_stop_time = copy.copy(full_state.time)
            
            passenger_waiting = full_state.stops[curr_stop_id].passenger_waiting
            if passenger_waiting == None:
                passenger_waiting = {}
            passenger_waiting[_new_time] = {'ons':curr_stop_ons, 'load':curr_stop_load, 'boarded':0, 'bLeft': False}
            full_state.stops[curr_stop_id].passenger_waiting = passenger_waiting
            
            log(self.logger, _new_time, f"Stop @ {curr_stop_id}: ons:{curr_stop_ons}, load:{curr_stop_load}", LogType.DEBUG)
            
            # TODO: Add events for when people leave
            return []
        
        if curr_event.event_type == EventType.PASSENGER_LEAVE_STOP:
            additional_info = curr_event.type_specific_information
            curr_stop_id = additional_info['stop_id']
            time_key = additional_info['time']
            
            # left_behind = additional_info['left_behind']
            passenger_waiting = full_state.stops[curr_stop_id].passenger_waiting
            
            ons = passenger_waiting[time_key]['ons']
            boarded = passenger_waiting[time_key]['boarded']
            
            left_behind = passenger_waiting[time_key].get('left_behind', 0)
            passenger_waiting[time_key]['left_behind'] = 0
            passenger_waiting[time_key]['bLeft'] = True
            passenger_waiting[time_key]['went_home']   = ons - boarded + left_behind
                
            full_state.stops[curr_stop_id].passenger_waiting = passenger_waiting
            log(self.logger, _new_time, f"Stop @ {curr_stop_id}: people left after {PASSENGER_TIME_TO_LEAVE} minutes.", LogType.ERROR)
            return []