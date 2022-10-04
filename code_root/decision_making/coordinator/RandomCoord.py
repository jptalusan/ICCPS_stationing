from src.utils import *
from Environment.enums import EventType, BusStatus, LogType, BusType, ActionType
from Environment.DataStructures.Event import Event
import random

# Limit dispatch to 1 overload bus per trip/route_id_dir
# Allocate to depot and activate buses
class RandomCoord:
    
    def __init__(self, 
                 environment_model,
                 travel_model,
                 dispatch_policy,
                 logger) -> None:
        self.environment_model = environment_model
        self.travel_model = travel_model
        self.dispatch_policy = dispatch_policy
        self.event_counter = 0
        self.logger = logger
        
        self.served_trips = []
        self.served_buses = []
        
        self.metrics = dict()
        self.metrics['resp_times'] = dict()
        self.metrics['computation_times'] = dict()
        
        self.random_seed = random.seed(102)
        pass
    
    def event_processing_callback_funct(self, actions, state):
        '''
        function that is called when each new event occurs in the underlying simulation.
        :param state:
        :param curr_event:
        :return: list of Events
        '''
        if len(actions) == 0:
            return []
        
        random_action = random.choice(actions)
        new_events = self.take_action(random_action, state)
        return new_events
    
    def add_incident(self, state, incident_event):
        incident = incident_event.type_specific_information['incident_obj']
        self.environment_model.add_incident(state, incident)
        
    def take_action(self, action, state):
        # print("take_action")
        action_type = action[0]
        ofb_id      = action[1]
        # ('HILLOMNF', '7_TO DOWNTOWN', datetime.datetime(2021, 8, 23, 14, 13, 11), 8.0, ('5692', '246343'))

        new_events = []
        
        # TODO: Make this better looking
        # Send to stop
        
        if ActionType.OVERLOAD_DISPATCH == action_type:
            ofb_obj = state.buses[ofb_id]
            
            action_info = action[2]
            stop_id = action_info[0]
            route_id_dir = action_info[1]
            arrival_time = action_info[2]
            remaining = action_info[3]
            current_block_trip = action_info[4]
            
            if current_block_trip in self.served_trips:
                return []
            
            log(self.logger, state.time, f"Taking random action: {action}, .", LogType.INFO)
            stop_no = self.travel_model.get_stop_number_at_id(current_block_trip, stop_id)
            
            ofb_obj.bus_block_trips = [current_block_trip]
            # Because at this point we already set the state to the next stop.
            ofb_obj.current_stop_number = stop_no
            ofb_obj.t_state_change = state.time
            
            event = Event(event_type=EventType.VEHICLE_START_TRIP, 
                          time=state.time)
            
            new_events.append(event)
            
            self.served_trips.append(current_block_trip)
        
        # Take over broken bus
        elif ActionType.OVERLOAD_TO_BROKEN == action_type:
            ofb_obj = state.buses[ofb_id]
            action_info = action[2]
            broken_bus_id = action_info
            
            if broken_bus_id in self.served_buses:
                return []
            
            log(self.logger, state.time, f"Taking random action: {action}, .", LogType.INFO)
            broken_bus_obj = state.buses[broken_bus_id]
            
            current_block_trip = broken_bus_obj.current_block_trip
            stop_no            = broken_bus_obj.current_stop_number
            
            ofb_obj.bus_block_trips = [broken_bus_obj.current_block_trip] + broken_bus_obj.bus_block_trips
            ofb_obj.current_block_trip = None
            # Because at this point we already set the state to the next stop.
            ofb_obj.current_stop_number = stop_no - 1
            ofb_obj.t_state_change = state.time
            
            # Switch passengers
            ofb_obj.current_load = broken_bus_obj.current_load
            ofb_obj.total_passengers_served = ofb_obj.current_load
            broken_bus_obj.current_load = 0
            
            event = Event(event_type=EventType.VEHICLE_START_TRIP, 
                          time=state.time)
            new_events.append(event)
            
            self.served_buses.append(broken_bus_id)
            log(self.logger, state.time, f"Sending takeover overflow bus: {ofb_id} to {broken_bus_obj.current_block_trip} @ stop {broken_bus_obj.current_stop}", LogType.ERROR)
        
        elif ActionType.OVERLOAD_ALLOCATE == action_type:
            # print(f"Random Coord: {action}")
            ofb_obj           = state.buses[ofb_id]
            current_stop      = ofb_obj.current_stop
            action_info       = action[2]
            reallocation_stop = action_info
            
            travel_time            = self.travel_model.get_travel_time_from_stop_to_stop(current_stop, reallocation_stop, state.time)
            distance_to_next_stop  = self.travel_model.get_distance_from_stop_to_stop(current_stop, reallocation_stop, state.time)
    
            ofb_obj.current_stop          = reallocation_stop
            ofb_obj.t_state_change        = state.time + dt.timedelta(seconds=travel_time)
            ofb_obj.status                = BusStatus.ALLOCATION
            ofb_obj.time_at_last_stop     = state.time
            ofb_obj.distance_to_next_stop = distance_to_next_stop
            
            event = Event(event_type=EventType.VEHICLE_START_TRIP, 
                          time=state.time)
            new_events.append(event)
            # new_events = self.dispatch_policy.
            log(self.logger, state.time, f"Reallocating overflow bus: {ofb_id} from {current_stop} to {reallocation_stop}", LogType.INFO)
            
        return new_events