
    
from Environment.enums import EventType


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
            new_events = []
            return new_events
        
        elif curr_event.event_type == EventType.PASSENGER_ARRIVE_STOP:
            new_events = []
            return new_events
        
        elif curr_event.event_type == EventType.PASSENGER_LEAVE_STOP:
            new_events = []
            return new_events
        
        elif curr_event.event_type == EventType.CONGESTION_LEVEL_CHANGE:
            new_events = []
            return new_events
        