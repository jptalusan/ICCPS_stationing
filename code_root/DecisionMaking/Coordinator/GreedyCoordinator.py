from Environment.enums import BusStatus, BusType, ActionType, EventType
from src.utils import *
import itertools
import random
import copy

class GreedyCoordinator:
    def __init__(self, 
                 travel_model,
                 dispatch_policy):
        self.travel_model = travel_model
        self.dispatch_policy = dispatch_policy
        self.trips_already_covered = []
        self.broken_buses_covered = []

    def event_processing_callback_funct(self, actions, state, action_type):
        # valid_actions = self.generate_possible_actions(state, action_type)
        valid_actions = self.generate_possible_actions(state, state.bus_events[0], action_type)
        # action_to_take = self.dispatch_policy.select_overload_to_dispatch(state, valid_actions)
        action_to_take = self.select_overload_to_dispatch(state, valid_actions)
        if action_to_take['type'] == ActionType.OVERLOAD_DISPATCH:
            self.trips_already_covered.append(action_to_take['info'][4])
        # if action_to_take['type'] == ActionType.OVERLOAD_TO_BROKEN:
        #     self.broken_buses_covered.append(action_to_take['info'][4])
        
        return action_to_take
    
    def generate_possible_actions_OLD(self, state, action_type=ActionType.OVERLOAD_ALL):
        # Find idle overload buses
        idle_overload_buses = []
        for bus_id, bus_obj in state.buses.items():
            if (bus_obj.type == BusType.OVERLOAD) and \
                    ((bus_obj.status == BusStatus.IDLE)
                     or (bus_obj.status == BusStatus.ALLOCATION)
                    ):
                # Prevent overload from being used when IDLE but has TRIPS left...
                if len(bus_obj.bus_block_trips) <= 0:
                    idle_overload_buses.append(bus_id)

        valid_actions = []
        if len(idle_overload_buses) <= 0:
            valid_actions = [{'type': ActionType.NO_ACTION, 'overload_bus': None, 'info': None}]
            return valid_actions
        
        if action_type == ActionType.OVERLOAD_ALLOCATE:
            _valid_actions = self.get_valid_allocations(state)
            valid_actions.extend(_valid_actions)
        elif action_type == ActionType.OVERLOAD_DISPATCH:
            stops_with_left_behind_passengers = []
            for stop_id, stop_obj in state.stops.items():
                passenger_waiting = stop_obj.passenger_waiting
                if not passenger_waiting:
                    continue

                for route_id_dir, route_pw in passenger_waiting.items():
                    if not route_pw:
                        continue

                    for arrival_time, pw in route_pw.items():
                        remaining_passengers = pw['remaining']
                        block_trip = pw['block_trip']

                        if remaining_passengers > 0:
                            # if block_trip not in self.trips_already_covered:
                            stops_with_left_behind_passengers.append((stop_id,
                                                                    route_id_dir,
                                                                    arrival_time,
                                                                    remaining_passengers,
                                                                    block_trip))

                _valid_actions = [[ActionType.OVERLOAD_DISPATCH], idle_overload_buses,
                                  stops_with_left_behind_passengers]
                _valid_actions = list(itertools.product(*_valid_actions))
                valid_actions.extend(_valid_actions)
        # elif action_type == ActionType.OVERLOAD_TO_BROKEN:
            broken_buses = []
            for bus_id, bus_obj in state.buses.items():
                if bus_obj.status == BusStatus.BROKEN:
                    if bus_obj.current_block_trip is not None:
                        broken_buses.append(bus_id)

            if len(broken_buses) > 0:
                _valid_actions = [[ActionType.OVERLOAD_TO_BROKEN], idle_overload_buses, broken_buses]
                _valid_actions = list(itertools.product(*_valid_actions))
                valid_actions.extend(_valid_actions)

        do_nothing_action = {'type': ActionType.NO_ACTION, 'overload_bus': None, 'info': None}
        if len(valid_actions) > 0:
            valid_actions = [{'type': _va[0], 'overload_bus': _va[1], 'info': _va[2]} for _va in valid_actions]
        else:
            # No action
            valid_actions = [do_nothing_action]
            
        return valid_actions

    def generate_possible_actions(self, state, event, action_type=ActionType.OVERLOAD_ALL):
        # Find idle overload buses
        idle_overload_buses = []
        for bus_id, bus_obj in state.buses.items():
            if (bus_obj.type == BusType.OVERLOAD) and \
                    ((bus_obj.status == BusStatus.IDLE)
                     or (bus_obj.status == BusStatus.ALLOCATION)
                    ):
                # Prevent overload from being used when IDLE but has TRIPS left...
                if len(bus_obj.bus_block_trips) <= 0:
                    idle_overload_buses.append(bus_id)

        if len(idle_overload_buses) <= 0:
            valid_actions = [{'type': ActionType.NO_ACTION, 'overload_bus': None, 'info': None}]
            return valid_actions

        # Create matrix of overload buses, original bus id, block/trips, stop_id
        valid_actions = []

        if action_type == ActionType.OVERLOAD_ALLOCATE or action_type == ActionType.OVERLOAD_ALL:
            _valid_actions = self.get_valid_allocations(state)
            valid_actions.extend(_valid_actions)

        if action_type == ActionType.OVERLOAD_DISPATCH or action_type == ActionType.OVERLOAD_ALL:
            if event.event_type == EventType.VEHICLE_ARRIVE_AT_STOP or \
                    event.event_type == EventType.DECISION_DISPATCH_EVENT or \
                    event.event_type == EventType.PASSENGER_LEFT_BEHIND:
                bus_id = event.type_specific_information['bus_id']
                if state.buses[bus_id].type != BusType.OVERLOAD and state.buses[bus_id].status == BusStatus.IN_TRANSIT:
                    if 'current_block_trip' in event.type_specific_information:
                        current_block_trip = event.type_specific_information['current_block_trip']
                        current_stop_number = event.type_specific_information['stop']
                    else:
                        current_block_trip = state.buses[bus_id].current_block_trip
                        current_stop_number = state.buses[bus_id].current_stop_number

                    past_stops = self.travel_model.get_list_of_stops_for_trip(current_block_trip[1], current_stop_number)
                    stops_with_left_behind_passengers = []

                    for stop_id in past_stops:
                        stop_obj = state.stops[stop_id]
                        passenger_waiting = stop_obj.passenger_waiting
                        if not passenger_waiting:
                            continue

                        for route_id_dir, route_pw in passenger_waiting.items():
                            if not route_pw:
                                continue

                            for arrival_time, pw in route_pw.items():
                                remaining_passengers = pw['remaining']
                                block_trip = pw['block_trip']

                                if remaining_passengers > 0:
                                    if block_trip not in self.trips_already_covered:
                                        stops_with_left_behind_passengers.append((stop_id,
                                                                                    route_id_dir,
                                                                                    arrival_time,
                                                                                    remaining_passengers,
                                                                                    block_trip))

                    # If no left behind people, just add current stop visited to actions space
                    if len(stops_with_left_behind_passengers) <= 0:
                        # if state.time >= state.buses[bus_id].t_state_change:
                        scheduled_arrival_time = self.travel_model.get_scheduled_arrival_time(current_block_trip, current_stop_number)
                        route_id_dir = self.travel_model.get_route_id_dir_for_trip(current_block_trip)
                        stops_with_left_behind_passengers.append((past_stops[-1],
                                                                    route_id_dir,
                                                                    scheduled_arrival_time,
                                                                    0,
                                                                    current_block_trip))

                    _valid_actions = [[ActionType.OVERLOAD_DISPATCH], idle_overload_buses,
                                        stops_with_left_behind_passengers]
                    _valid_actions = list(itertools.product(*_valid_actions))
                    valid_actions.extend(_valid_actions)

        broken_buses = []
        for bus_id, bus_obj in state.buses.items():
            if bus_obj.status == BusStatus.BROKEN:
                # Without checking if a broken bus has already been covered, we try to cover it again
                # Leading to null values
                if bus_obj.current_block_trip is not None:
                    broken_buses.append(bus_id)
            pass

        if len(broken_buses) > 0:
            _valid_actions = [[ActionType.OVERLOAD_TO_BROKEN], idle_overload_buses, broken_buses]
            _valid_actions = list(itertools.product(*_valid_actions))
            valid_actions.extend(_valid_actions)

        # Constraint on broken bus
        if len(broken_buses) > 0:
        # if event and (event.event_type == EventType.VEHICLE_BREAKDOWN):
            constrained_combo_actions = []
            for action in valid_actions:
                _action_type = action['type']
                if _action_type == ActionType.OVERLOAD_TO_BROKEN:
                    if action not in constrained_combo_actions:
                        constrained_combo_actions.append(action)
            valid_actions = copy.copy(constrained_combo_actions)

        do_nothing_action = {'type': ActionType.NO_ACTION, 'overload_bus': None, 'info': None}
        if len(valid_actions) > 0:
            valid_actions = [{'type': _va[0], 'overload_bus': _va[1], 'info': _va[2]} for _va in valid_actions]
        else:
            # No action
            valid_actions = [do_nothing_action]

        return valid_actions
    
    def get_valid_allocations(self, state):
        num_available_buses = len(
            [_ for _ in state.buses.values() if _.status == BusStatus.IDLE and _.type == BusType.OVERLOAD])

        if num_available_buses <= 0:
            return []

        # MTA, MCC5_1, HICHICNN, WESWILEN (based on MTA)
        valid_stops = ['MTA', 'MCC5_1', 'HICHICNN', 'WESWILEN']

        # Based on spatial clustering k = 10
        idle_overload_buses = []
        for bus_id, bus_obj in state.buses.items():
            if (bus_obj.type == BusType.OVERLOAD) and \
                    ((bus_obj.status == BusStatus.IDLE)
                     or (bus_obj.status == BusStatus.ALLOCATION)
                    ):
                if len(bus_obj.bus_block_trips) <= 0:
                    idle_overload_buses.append(bus_id)

        valid_actions = []
        _valid_actions = [[ActionType.OVERLOAD_ALLOCATE], idle_overload_buses, valid_stops]
        _valid_actions = list(itertools.product(*_valid_actions))

        # Remove allocations to the same stop the bus is currently in.
        _valid_actions = [va for va in _valid_actions if va[2] != state.buses[va[1]].current_stop]
        valid_actions.extend(_valid_actions)
        return valid_actions
    
    def select_overload_to_dispatch(self, state, actions):
        random.seed(100)
        if len(actions) <= 0:
            return None

        actions_with_distance = []

        is_all_allocation = True
        for action in actions:
            if action['type'] != ActionType.OVERLOAD_ALLOCATE and action['type'] != ActionType.NO_ACTION:
                is_all_allocation = False
                
        if is_all_allocation:
            return random.choice(actions)
    
        for action in actions:
            action_type = action['type']
            overload_bus = action['overload_bus']
            info = action['info']

            if (action_type == ActionType.NO_ACTION) and (len(actions) == 1):
                return actions[0]

            elif (action_type == ActionType.NO_ACTION) and (len(actions) > 1):
                continue
            
            current_stop = state.buses[overload_bus].current_stop
            next_stop = None

            if action_type == ActionType.OVERLOAD_TO_BROKEN:
                broken_bus = info
                next_stop = state.buses[info].current_stop
            elif action_type == ActionType.OVERLOAD_DISPATCH:
                next_stop = info[0]
                pass
            elif action_type == ActionType.OVERLOAD_ALLOCATE:
                next_stop = info
                pass
            else:
                raise "Action not supported"

            distance = self.travel_model.get_distance_from_stop_to_stop(current_stop, next_stop, state.time)
            actions_with_distance.append((action, distance))
            
        if len(actions_with_distance) == 0:
            action_to_take = {'type': ActionType.NO_ACTION, 'overload_bus': None, 'info': None}
            return action_to_take
        
        actions_with_distance = sorted(actions_with_distance, key=lambda x: x[1], reverse=False)
        # print(actions_with_distance)
        actions_with_distance = actions_with_distance[0][0]
        return actions_with_distance