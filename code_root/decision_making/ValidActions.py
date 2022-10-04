import copy
import itertools
from src.utils import *

from Environment.enums import BusStatus, BusType, ActionType

'''
Should be tied with environment model since valid actions are a direct consequence of the current environment
'''
class ValidActions:
    
    def __init__(self, travel_model, logger) -> None:
        self.travel_model = travel_model
        self.logger = logger
        self.served_trips = []
        pass

    # Get passengers which are left behind, if can fetch them before they leave. Then do so.
    def get_valid_actions(self, state):
        _state = copy.copy(state)
        
        num_available_buses = len([_ for _ in state.buses.values() if _.status == BusStatus.IDLE and _.type == BusType.OVERLOAD])
        if num_available_buses <= 0:
            return []
        
        # Passengers left behind (must identify the trips that cover these stops)
        stops_with_left_behind_passengers = []
        for stop_id, stop_obj in _state.stops.items():
            passenger_waiting = stop_obj.passenger_waiting
            if not passenger_waiting:
                continue
            
            for route_id_dir, route_pw in passenger_waiting.items():
                if not route_pw:
                    continue
                
                for arrival_time, pw in route_pw.items():
                    remaining_passengers = pw['remaining']
                    block_trip = pw['block_trip']
                    
                    if block_trip in self.served_trips:
                        continue
                    
                    if remaining_passengers > 0:
                        stops_with_left_behind_passengers.append((stop_id, route_id_dir, arrival_time, remaining_passengers, block_trip))
                        self.served_trips.append(block_trip)
        
        # Find broken buses
        broken_buses = []
        for bus_id, bus_obj in _state.buses.items():
            if bus_obj.status == BusStatus.BROKEN:
                # log(self.logger, _state.time, f"Found broken bus: {bus_id}.")
                broken_buses.append(bus_id)
            pass
        
        # Find idle overload buses
        idle_overload_buses = []
        for bus_id, bus_obj in _state.buses.items():
            if bus_obj.type == BusType.OVERLOAD:
                idle_overload_buses.append(bus_id)

        # Create matrix of overload buses, original bus id, block/trips, stop_id
        valid_actions = []
        
        _valid_actions = [[ActionType.OVERLOAD_DISPATCH], idle_overload_buses, stops_with_left_behind_passengers]
        _valid_actions = list(itertools.product(*_valid_actions))
        valid_actions.extend(_valid_actions)
        
        # print(broken_buses)
        _valid_actions = [[ActionType.OVERLOAD_TO_BROKEN], idle_overload_buses, broken_buses]
        _valid_actions = list(itertools.product(*_valid_actions))
        valid_actions.extend(_valid_actions)
        
        # _valid_actions = self.get_valid_allocations(state)
        # valid_actions.extend(_valid_actions)
        
        # print("Number of valid actions:", len(valid_actions))
        if len(valid_actions) > 0:
            # print(state.time)
            # print(valid_actions)
            # print()
            pass
        return valid_actions

    def get_valid_allocations(self, state):
        num_available_buses = len([_ for _ in state.buses.values() if _.status == BusStatus.IDLE and _.type == BusType.OVERLOAD])
        if num_available_buses <= 0:
            return []
        
        valid_stops = list(state.stops.keys())
        
        idle_overload_buses = []
        for bus_id, bus_obj in state.buses.items():
            if bus_obj.type == BusType.OVERLOAD and bus_obj.status == BusStatus.IDLE:
                idle_overload_buses.append(bus_id)
        
        valid_actions = []
        _valid_actions = [[ActionType.OVERLOAD_ALLOCATE], idle_overload_buses, valid_stops]
        _valid_actions = list(itertools.product(*_valid_actions))
        valid_actions.extend(_valid_actions)
        return valid_actions