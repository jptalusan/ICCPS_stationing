from src.utils import *
from Environment.enums import EventType, BusStatus, LogType, BusType
from Environment.DataStructures.Event import Event
import random

# Limit dispatch to 1 overload bus per trip/route_id_dir
# Allocate to depot and activate buses
class DispatchOnlyCoord:
    
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
        
        self.metrics = dict()
        self.metrics['resp_times'] = dict()
        self.metrics['computation_times'] = dict()
        
        self.random_seed = random.seed(100)
        pass
    
    def event_processing_callback_funct(self, state, curr_event):
        '''
        function that is called when each new event occurs in the underlying simulation.
        :param state:
        :param curr_event:
        :return: list of Events
        '''
        
        if curr_event.event_type == EventType.VEHICLE_BREAKDOWN:
            type_specific_information = curr_event.type_specific_information
            bus_id = type_specific_information['bus_id']
            bus_obj = state.buses[bus_id]
            bus_obj.status = BusStatus.BROKEN
            
            new_events = self.dispatch_overload_buses_takeover(state, curr_event)
            return new_events
        
        elif curr_event.event_type == EventType.VEHICLE_ACCIDENT:
            new_events = []
            return new_events
        
        # Check if passengers are left behind.
        elif curr_event.event_type == EventType.VEHICLE_ARRIVE_AT_STOP:
            new_events = []
            _new_events = self.dispatch_overload_buses(state, curr_event)
            new_events.extend(_new_events)
            # _new_events = self.move_idle_overload_buses(state, curr_event)
            # new_events.extend(_new_events)
            
            return new_events
        
        return []
    
    def add_incident(self, state, incident_event):
        incident = incident_event.type_specific_information['incident_obj']
        self.environment_model.add_incident(state, incident)
        
    # This is the action_taker. Taking the actions based on the decisions of the decision maker (which does not update anything).
    def dispatch_overload_buses_takeover(self, state, curr_event):
        new_events = []
        num_available_buses = len([_ for _ in state.buses.values() if _.status == BusStatus.IDLE])
        if num_available_buses <= 0:
            return []
        
        # Only look at current stop of the event
        bus_id  = curr_event.type_specific_information['bus_id']
        bus_obj = state.buses[bus_id]
        
        current_block_trip = bus_obj.current_block_trip
        stop_no            = bus_obj.current_stop_number
        
        stop_id = self.travel_model.get_stop_id_at_number(current_block_trip, stop_no)
        
        log(self.logger, state.time, f"Bus {bus_id} broke down after stop {bus_obj.current_stop_number}:{bus_obj.current_stop}", LogType.ERROR)
        
        ofb_id = self.dispatch_policy.get_overflow_bus_to_overflow_stop(state)
        if ofb_id:
            ofb_obj = state.buses[ofb_id]
            ofb_obj.bus_block_trips = [bus_obj.current_block_trip] + bus_obj.bus_block_trips
            ofb_obj.current_block_trip = None
            # Because at this point we already set the state to the next stop.
            ofb_obj.current_stop_number = stop_no - 1
            ofb_obj.t_state_change = curr_event.time
            
            ofb_obj.current_load = bus_obj.current_load
            bus_obj.current_load = 0
            
            event = Event(event_type=EventType.VEHICLE_START_TRIP, 
                            time=curr_event.time)
            new_events.append(event)
            
            self.served_trips.append(current_block_trip)
            
            log(self.logger, state.time, f"Sending takeover overflow bus: {ofb_id} to {bus_obj.current_block_trip} @ stop {bus_obj.current_stop}", LogType.ERROR)
        return new_events
    
    def move_idle_overload_buses(self, state, curr_event):
        new_events = []
        num_available_buses = len([_ for _ in state.buses.values() if _.status == BusStatus.IDLE and _.type == BusType.OVERLOAD])
        if num_available_buses <= 0:
            return []
        
        stop_list = list(state.stops.keys())
        
        for bus_id, bus_obj in state.buses.items():
            if bus_obj.type == BusType.OVERLOAD:
                if bus_obj.status == BusStatus.IDLE:
                    current_stop = bus_obj.current_stop
                    reallocation = random.choice(stop_list)
                    
                    travel_time            = self.travel_model.get_travel_time_from_stop_to_stop(current_stop, reallocation, curr_event.time)
                    distance_to_next_stop  = self.travel_model.get_distance_from_stop_to_stop(current_stop, reallocation, curr_event.time)
                    bus_obj.current_stop   = reallocation
                    bus_obj.t_state_change = curr_event.time + dt.timedelta(seconds=travel_time)
                    bus_obj.status = BusStatus.ALLOCATION
                    bus_obj.time_at_last_stop = curr_event.time
                    bus_obj.distance_to_next_stop = distance_to_next_stop
                    
                    event = Event(event_type=EventType.VEHICLE_START_TRIP, 
                                  time=curr_event.time)
                    new_events.append(event)
                    log(self.logger, curr_event.time, f"Reallocating overflow bus: {bus_id} from {current_stop} to {reallocation}", LogType.INFO)
        return new_events
    
    # This is the action_taker. Taking the actions based on the decisions of the decision maker (which does not update anything).
    def dispatch_overload_buses(self, state, curr_event):
        log(self.logger, curr_event.time, f"Dispatch_overload: {curr_event}.")
        new_events = []
        num_available_buses = len([_ for _ in state.buses.values() if _.status == BusStatus.IDLE and _.type == BusType.OVERLOAD])
        if num_available_buses <= 0:
            return []
        
        # Only look at current stop of the event
        bus_id = curr_event.type_specific_information['bus_id']
        current_block_trip = curr_event.type_specific_information['current_block_trip']
        bus_obj = state.buses[bus_id]
        stop_no = curr_event.type_specific_information['stop']
        
        last_stop_of_trip = self.travel_model.get_last_stop_number_on_trip(current_block_trip)
        if stop_no == last_stop_of_trip:
            return []
        
        if current_block_trip in self.served_trips:
            return []
        
        stop_id = self.travel_model.get_stop_id_at_number(current_block_trip, stop_no)
        stop_obj = state.stops[stop_id]
        route_id_dir = self.travel_model.get_route_id_dir_for_trip(current_block_trip)
        
        passenger_waiting = stop_obj.passenger_waiting
        if passenger_waiting:
            remaining = 0
            for _route_id_dir, _p in passenger_waiting.items():
                log(self.logger, curr_event.time, f"Dispatch: {stop_id}, {_route_id_dir}, {route_id_dir}, {_p}")
                if (route_id_dir == _route_id_dir) & len(_p) > 0:
                    for _datetime, passengers in _p.items():
                        if len(passengers) > 0:
                            remaining = passengers['remaining']

            if remaining:
                ofb_id = self.dispatch_policy.get_overflow_bus_to_overflow_stop(state)
                if ofb_id:
                    ofb_obj = state.buses[ofb_id]
                    ofb_obj.bus_block_trips = [current_block_trip]
                    # Because at this point we already set the state to the next stop.
                    ofb_obj.current_stop_number = stop_no
                    ofb_obj.t_state_change = curr_event.time
                    
                    event = Event(event_type=EventType.VEHICLE_START_TRIP, 
                                  time=curr_event.time)
                    new_events.append(event)
                    
                    self.served_trips.append(current_block_trip)
                    
                    log(self.logger, curr_event.time, f"Sending overflow bus: {ofb_id} to {bus_obj.current_block_trip} @ stop {stop_id}", LogType.ERROR)
        return new_events