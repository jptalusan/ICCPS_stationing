from Environment.EnvironmentModel import EnvironmentModel
from Environment.enums import BusStatus, BusType, ActionType, LogType
from src.utils import *
from Environment.DataStructures.Event import Event
from Environment.enums import EventType
import itertools
import copy
import datetime as dt

# Should have get possible actions function


class DecisionEnvironmentDynamics(EnvironmentModel):

    def __init__(self, travel_model, send_nearest_dispatch, reward_policy=None, logger=None):
        EnvironmentModel.__init__(self, travel_model=travel_model, logger=logger)
        self.send_nearest_dispatch = send_nearest_dispatch
        self.reward_policy         = reward_policy
        self.travel_model          = travel_model
        self.logger                = logger
        # self.trips_already_covered = []
        # self.served_buses          = []

    # Have to include "NO ACTION"
    def generate_possible_actions(self, state, event, action_type=ActionType.OVERLOAD_ALL):
        num_available_buses = len(
            [_ for _ in state.buses.values() if _.status == BusStatus.IDLE and _.type == BusType.OVERLOAD])

        if num_available_buses <= 0:
            valid_actions = [{'type': ActionType.NO_ACTION, 'overload_bus': None, 'info': None}]
            action_taken_tracker = [(_[0], False) for _ in enumerate(valid_actions)]
            return valid_actions, action_taken_tracker

        # Passengers left behind (must identify the trips that cover these stops)
        stops_with_left_behind_passengers = []
        total_remaining = 0
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

                    # if block_trip in self.trips_already_covered:
                    #     continue

                    if remaining_passengers > 0:
                        stops_with_left_behind_passengers.append((stop_id,
                                                                  route_id_dir,
                                                                  arrival_time,
                                                                  remaining_passengers,
                                                                  block_trip))
                        # self.trips_already_covered.append(block_trip)
                        total_remaining += remaining_passengers

        # Find broken buses
        broken_buses = []
        for bus_id, bus_obj in state.buses.items():
            if bus_obj.status == BusStatus.BROKEN:
                # log(self.logger, _state.time, f"Found broken bus: {bus_id}.")
                broken_buses.append(bus_id)
            pass

        # Find idle overload buses
        idle_overload_buses = []
        for bus_id, bus_obj in state.buses.items():
            if bus_obj.type == BusType.OVERLOAD:
                idle_overload_buses.append(bus_id)

        # Create matrix of overload buses, original bus id, block/trips, stop_id
        valid_actions = []

        # print(f"mdpEnv::Remaining people: {total_remaining}")
        # print(f"mdpEnv::Stops with remaining people: {stops_with_left_behind_passengers}")
        # Dispatch
        if action_type == ActionType.OVERLOAD_ALL:
            _valid_actions = [[ActionType.OVERLOAD_DISPATCH], idle_overload_buses, stops_with_left_behind_passengers]
            _valid_actions = list(itertools.product(*_valid_actions))
            valid_actions.extend(_valid_actions)

            _valid_actions = [[ActionType.OVERLOAD_TO_BROKEN], idle_overload_buses, broken_buses]
            _valid_actions = list(itertools.product(*_valid_actions))
            valid_actions.extend(_valid_actions)

            # # Allocation
            # _valid_actions = self.get_valid_allocations(state)
            # valid_actions.extend(_valid_actions)

        elif action_type == ActionType.OVERLOAD_TO_BROKEN:
            _valid_actions = [[ActionType.OVERLOAD_TO_BROKEN], idle_overload_buses, broken_buses]
            _valid_actions = list(itertools.product(*_valid_actions))
            valid_actions.extend(_valid_actions)

        elif action_type == ActionType.OVERLOAD_DISPATCH:
            _valid_actions = [[ActionType.OVERLOAD_DISPATCH], idle_overload_buses, stops_with_left_behind_passengers]
            _valid_actions = list(itertools.product(*_valid_actions))
            valid_actions.extend(_valid_actions)

        elif action_type == ActionType.OVERLOAD_ALLOCATE:
            _valid_actions = self.get_valid_allocations(state)
            valid_actions.extend(_valid_actions)

        # print("mdpEnv::Number of valid actions:", len(valid_actions))
        # print("mdpEnv::Valid actions:", valid_actions)

        if len(valid_actions) > 0:
            valid_actions = [{'type': _va[0], 'overload_bus': _va[1], 'info': _va[2]} for _va in valid_actions]
        else:
            # No action
            valid_actions = [{'type': ActionType.NO_ACTION, 'overload_bus': None, 'info': None}]

        action_taken_tracker = [(_[0], False) for _ in enumerate(valid_actions)]
        return valid_actions, action_taken_tracker

    def get_valid_allocations(self, state):
        num_available_buses = len(
            [_ for _ in state.buses.values() if _.status == BusStatus.IDLE and _.type == BusType.OVERLOAD])

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

    def take_action(self, state, action):
        # print("take_action")
        action_type = action['type']
        ofb_id      = action['overload_bus']
        # ('HILLOMNF', '7_TO DOWNTOWN', datetime.datetime(2021, 8, 23, 14, 13, 11), 8.0, ('5692', '246343'))

        new_events = []

        # Send to stop

        if ActionType.OVERLOAD_DISPATCH == action_type:
            ofb_obj = state.buses[ofb_id]

            action_info        = action["info"]
            stop_id            = action_info[0]
            route_id_dir       = action_info[1]
            arrival_time       = action_info[2]
            remaining          = action_info[3]
            current_block_trip = action_info[4]

            # if current_block_trip in self.trips_already_covered:
            #     return []

            log(self.logger, state.time,
                f"Taking MCTS action: {action}, .",
                LogType.INFO)
            stop_no = self.travel_model.get_stop_number_at_id(current_block_trip, stop_id)

            ofb_obj.bus_block_trips = [current_block_trip]
            # Because at this point we already set the state to the next stop.
            ofb_obj.current_stop_number = stop_no
            ofb_obj.t_state_change = state.time

            event = Event(event_type=EventType.VEHICLE_START_TRIP,
                          time=state.time + dt.timedelta(seconds=1))

            new_events.append(event)

            # self.trips_already_covered.append(current_block_trip)

        # Take over broken bus
        elif ActionType.OVERLOAD_TO_BROKEN == action_type:
            ofb_obj       = state.buses[ofb_id]
            action_info   = action["info"]
            broken_bus_id = action_info

            # if broken_bus_id in self.served_buses:
            #     return []

            log(self.logger, state.time,
                f"Taking MCTS action: {action}, .",
                LogType.INFO)
            broken_bus_obj = state.buses[broken_bus_id]

            current_block_trip = broken_bus_obj.current_block_trip
            stop_no            = broken_bus_obj.current_stop_number

            ofb_obj.bus_block_trips = copy.copy([broken_bus_obj.current_block_trip] + broken_bus_obj.bus_block_trips)
            ofb_obj.current_block_trip = None
            # Because at this point we already set the state to the next stop.
            ofb_obj.current_stop_number = stop_no - 1
            ofb_obj.t_state_change = state.time + dt.timedelta(seconds=1)

            # Switch passengers
            ofb_obj.current_load = copy.copy(broken_bus_obj.current_load)
            ofb_obj.total_passengers_served = ofb_obj.current_load

            # Deactivate broken_bus_obj
            # broken_bus_obj.total_passengers_served -= broken_bus_obj.current_load
            broken_bus_obj.current_load = 0
            broken_bus_obj.current_block_trip = None
            broken_bus_obj.bus_block_trips = []

            event = Event(event_type=EventType.VEHICLE_START_TRIP,
                          time=state.time + dt.timedelta(seconds=1))
            new_events.append(event)

            # self.served_buses.append(broken_bus_id)
            log(self.logger, state.time,
                f"Sending takeover overflow bus: {ofb_id} to {broken_bus_obj.current_block_trip} @ stop {broken_bus_obj.current_stop}",
                LogType.ERROR)

        elif ActionType.OVERLOAD_ALLOCATE == action_type:
            # print(f"Random Coord: {action}")
            ofb_obj = state.buses[ofb_id]
            current_stop = ofb_obj.current_stop
            action_info = action["info"]
            reallocation_stop = action_info

            travel_time = self.travel_model.get_travel_time_from_stop_to_stop(current_stop,
                                                                              reallocation_stop,
                                                                              state.time)

            distance_to_next_stop = self.travel_model.get_distance_from_stop_to_stop(current_stop,
                                                                                     reallocation_stop,
                                                                                     state.time)

            ofb_obj.current_stop = reallocation_stop
            ofb_obj.t_state_change = state.time + dt.timedelta(seconds=travel_time)
            ofb_obj.status = BusStatus.ALLOCATION
            ofb_obj.time_at_last_stop = state.time
            ofb_obj.distance_to_next_stop = distance_to_next_stop

            event = Event(event_type=EventType.VEHICLE_START_TRIP,
                          time=state.time + dt.timedelta(seconds=1))
            new_events.append(event)
            # new_events = self.dispatch_policy.
            log(self.logger, state.time,
                f"Reallocating overflow bus: {ofb_id} from {current_stop} to {reallocation_stop}",
                LogType.INFO)

        elif ActionType.NO_ACTION == action_type:
            # Do nothing
            new_events = None
            pass

        reward = self.compute_reward(state)
        return reward, new_events, state.time

    def compute_reward(self, state):
        total_remaining = 0
        for stop_id, stop_obj in state.stops.items():
            passenger_waiting = stop_obj.passenger_waiting
            if not passenger_waiting:
                continue

            for route_id_dir, route_pw in passenger_waiting.items():
                if not route_pw:
                    continue

                for arrival_time, pw in route_pw.items():
                    remaining_passengers = pw['remaining']
                    total_remaining += remaining_passengers

        # print(f"mdpEnv::reward: {-1 * total_remaining}")
        return -1 * total_remaining