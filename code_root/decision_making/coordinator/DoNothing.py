from src.utils import *
from Environment.enums import EventType, BusStatus

class DoNothing:
    
    def __init__(self, 
                 environment_model,
                 travel_model,
                 dispatch_policy,
                 logger) -> None:
        self.environment_model = environment_model
        self.travel_model = travel_model
        self.dispatch_policy = dispatch_policy
        self.event_couter = 0
        self.logger = logger
        pass
    
    def event_processing_callback_funct(self, state, curr_event):
        
        self.event_couter += 1
        
        if curr_event.event_type == EventType.VEHICLE_START_TRIP:
            new_events = []
            return new_events
        
        elif curr_event.event_type == EventType.VEHICLE_ARRIVE_AT_STOP:
            new_events = []
            return new_events
        
        elif curr_event.event_type == EventType.VEHICLE_ACCIDENT:
            new_events = []
            return new_events
        
        elif curr_event.event_type == EventType.VEHICLE_BREAKDOWN:
            type_specific_information = curr_event.type_specific_information
            bus_id = type_specific_information['bus_id']
            bus_obj = state.buses[bus_id]
            current_block_trip = bus_obj.current_block_trip
            current_stop_number = bus_obj.current_stop_number
            current_stop_id = self.travel_model.get_stop_id_at_number(current_block_trip, current_stop_number)
            bus_obj.status = BusStatus.BROKEN
            
            log(self.logger, curr_event.time, f"Vehicle {bus_id} broke down at stop {current_stop_id}", LogType.DEBUG)
            new_events = []
            return new_events
        
        elif curr_event.event_type == EventType.PASSENGER_ARRIVE_STOP:
            new_events = []
            return new_events
        
        elif curr_event.event_type == EventType.PASSENGER_LEAVE_STOP:
            type_specific_information = curr_event.type_specific_information
            time_key = type_specific_information['time']
            stop_id = type_specific_information['stop_id']
            stop_obj = state.stops[stop_id]
            passenger_waiting = stop_obj.passenger_waiting
            bLeft = passenger_waiting[time_key]['bLeft']
            if bLeft:
                log(self.logger, curr_event.time, f"People left stop {stop_id}, setup action...", LogType.DEBUG)
            new_events = []
            return new_events
        
        elif curr_event.event_type == EventType.CONGESTION_LEVEL_CHANGE:
            new_events = []
            return new_events
        