import copy
import random

class StopDynamics:
    
    def __init__(self) -> None:
        pass
    
    def update_stop(self, curr_event, _new_time, full_state):
        additional_info = curr_event.type_specific_information
        curr_stop_id = additional_info['stop_id']
        curr_stop_ons = additional_info['ons']
        curr_stop_load = additional_info['load']
        
        curr_stop_time = copy.copy(full_state.time)
        
        passenger_waiting = full_state.stops[curr_stop_id].passenger_waiting
        if passenger_waiting == None:
            passenger_waiting = {}
        passenger_waiting[_new_time] = {'ons':curr_stop_ons}
        full_state.stops[curr_stop_id].passenger_waiting = passenger_waiting
        