import copy
import time
import json
import datetime as dt
import pandas as pd
from Environment.DataStructures.State import State
from Environment.DataStructures.Event import Event
from decision_making.CentralizedMCTS.DataStructures.TreeNode import TreeNode
from Environment.enums import EventType, ActionType, BusType, BusStatus
from src.utils import *


class BareMinimumRollout:
    """
    Bare minimum rollout, send the nearest bus (if available) to cover for a broken down bus.
    """

    def __init__(self):
        self.deep_copy_time = 0
        self.debug_rewards = []
        self.curr_event = None
        self.rollout_horizon_delta_t = 60 * 60 * 0.6  # 60*60*N for N hour horizon
        # self.rollout_horizon_delta_t = None

        config_path = 'scenarios/baseline/data/trip_plan.json'
        with open(config_path) as f:
            self.trip_plan = json.load(f)

        lookup_path = 'scenarios/baseline/data/pair_tt_dd_stops.pkl'
        self.lookup_tt_dd = pd.read_pickle(lookup_path)

        self.total_walkaways = 0

    def rollout(self,
                node,
                environment_model,
                discount_factor,
                solve_start_time):
        s_copy_time = time.time()
        self.debug_rewards = []
        self.curr_event = None
        self.total_walkaways = 0

        truncated_events = copy.copy(node.future_events_queue)

        if self.rollout_horizon_delta_t is not None:
            lookahead_horizon = node.state.time + dt.timedelta(seconds=self.rollout_horizon_delta_t)
            truncated_events = [event for event in truncated_events if
                                node.state.time <= event.time <= lookahead_horizon]

        _state = State(
            stops=copy.deepcopy(node.state.stops),
            buses=copy.deepcopy(node.state.buses),
            events=truncated_events,
            # events=copy.copy(node.future_events_queue),
            # events=copy.copy(node.state.events),
            time=node.state.time
        )

        # Why is possible_actions None?
        _node = TreeNode(
            state=_state,
            parent=None,
            depth=node.depth,
            is_terminal=node.is_terminal,
            possible_actions=None,
            action_to_get_here=None,
            score=node.score,
            num_visits=None,
            children=None,
            reward_to_here=node.reward_to_here,
            is_fully_expanded=False,
            actions_taken_tracker=None,
            action_sequence_to_here=None,
            event_at_node=node.event_at_node,
            # future_events_queue=copy.copy(node.future_events_queue)
            future_events_queue=truncated_events
        )

        self.deep_copy_time += time.time() - s_copy_time

        # Run until all events finish
        while _node.future_events_queue:
            self.rollout_iter(_node, environment_model, discount_factor, solve_start_time)

        # print(f"Total walk aways: {self.total_walkaways}")
        # [print(dr) for dr in self.debug_rewards if dr['type'] != ActionType.NO_ACTION]
        # print()
        return _node.reward_to_here

    """
    SendNearestDispatch if a vehicle is broken, else do nothing.
    """

    def rollout_iter(self, node, environment_model, discount_factor, solve_start_time):
        valid_actions, _ = environment_model.generate_possible_actions(node.state, None)

        # Send nearest dispatch
        action_to_take = environment_model.get_rollout_actions(node.state, valid_actions)
        if not action_to_take:
            action_to_take = {'type': ActionType.NO_ACTION, 'overload_bus': None, 'info': None}

        # Comment this and uncomment above if needs more complex rollout
        # action_to_take = {'type': ActionType.NO_ACTION, 'overload_bus': None, 'info': None}

        immediate_reward, new_events, event_time = environment_model.take_action(node.state, action_to_take)
        # NOTE: New events returns a list even though its only returning a single event always
        if new_events is not None:
            node.future_events_queue.append(new_events[0])
            node.future_events_queue.sort(key=lambda _: _.time)

        node.depth += 1
        node.event_at_node = node.future_events_queue.pop(0)
        # self.process_event(node.state, node.event_at_node, environment_model)
        new_events = self.fast_process_event(node.state, node.event_at_node)
        if len(new_events) > 0:
            node.future_events_queue.append(new_events[0])
            node.future_events_queue.sort(key=lambda _: _.time)

        discounted_immediate_score = self.standard_discounted_score(immediate_reward,
                                                                    event_time - solve_start_time,
                                                                    discount_factor)

        node.reward_to_here = node.reward_to_here + discounted_immediate_score

        self.debug_rewards.append(action_to_take)

    """
    - Plan, create a vector in state that is just [[stops][remainings] for passenger arrivals
    - basically reduce loops per rollout iteration
    - Use just vehicle_arrival and update vehicle location and where they will go (do last, not sure how possible)
    """

    def standard_discounted_score(self, reward, time_since_start, discount_factor):
        discount = discount_factor ** time_since_start.total_seconds()
        discounted_reward = discount * reward
        return discounted_reward

    def process_event(self, state, event, environment_model):
        environment_model.update(state, event)

    def fast_process_event(self, state, event):
        new_events = []
        if event.event_type == EventType.VEHICLE_BREAKDOWN:
            type_specific_information = event.type_specific_information
            event_bus_id = type_specific_information['bus_id']
            current_block_trip = state.buses[event_bus_id].current_block_trip
            state.buses[event_bus_id].status = BusStatus.BROKEN
            # state.buses[event_bus_id].t_state_change = state.time

        elif event.event_type == EventType.PASSENGER_ARRIVE_STOP:
            additional_info = event.type_specific_information
            arrival_time = event.time
            curr_route_id_dir = additional_info['route_id_dir']
            curr_stop_id = additional_info['stop_id']
            curr_stop_ons = additional_info['ons']
            curr_stop_offs = additional_info['offs']

            passenger_waiting = state.stops[curr_stop_id].passenger_waiting
            if passenger_waiting is None:
                passenger_waiting = {}

            # Initial values for the passenger dictionary
            passenger_waiting[curr_route_id_dir] = {}
            passenger_waiting[curr_route_id_dir][arrival_time] = {'got_on_bus': 0,
                                                                  'remaining': 0,
                                                                  'block_trip': "",
                                                                  'ons': curr_stop_ons,
                                                                  'offs': curr_stop_offs}

            state.stops[curr_stop_id].passenger_waiting = passenger_waiting

        elif event.event_type == EventType.PASSENGER_LEAVE_STOP:
            additional_info = event.type_specific_information
            curr_route_id_dir = additional_info['route_id_dir']
            curr_stop_id = additional_info['stop_id']
            time_key = additional_info['time']
            passenger_waiting = state.stops[curr_stop_id].passenger_waiting

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
                state.stops[curr_stop_id].total_passenger_walk_away += remaining

                # Delete dictionary for this time
                del state.stops[curr_stop_id].passenger_waiting[curr_route_id_dir][time_key]

            self.total_walkaways += state.stops[curr_stop_id].total_passenger_walk_away

        elif event.event_type == EventType.VEHICLE_START_TRIP:
            additional_info = event.type_specific_information
            bus_id = additional_info['bus_id']
            bus_state = state.buses[bus_id].status

            if BusStatus.IDLE == bus_state:
                if len(state.buses[bus_id].bus_block_trips) > 0:

                    state.buses[bus_id].status = BusStatus.IN_TRANSIT
                    current_block_trip = state.buses[bus_id].bus_block_trips.pop(0)
                    state.buses[bus_id].current_block_trip = current_block_trip
                    travel_time = 100
                    time_to_state_change = state.time + dt.timedelta(seconds=travel_time)

                    event = Event(event_type=EventType.VEHICLE_ARRIVE_AT_STOP,
                                  time=time_to_state_change,
                                  type_specific_information={'bus_id': bus_id,
                                                             'current_block_trip': current_block_trip,
                                                             'stop': state.buses[bus_id].current_stop_number})
                    new_events.append(event)
                # no more trips left
                else:
                    raise "Should not have an IN_TRANSIT bus starting a trip."
                    pass

            elif BusStatus.IN_TRANSIT == bus_state:
                # print(f"Bus {bus_id} is IN TRANSIT")
                pass

        elif event.event_type == EventType.VEHICLE_ARRIVE_AT_STOP:
            additional_info = event.type_specific_information
            bus_id = additional_info['bus_id']
            bus_state = state.buses[bus_id].status

            if BusStatus.IDLE == bus_state:
                raise "Should not have an IDLE bus arriving at a stop."
                pass

            elif BusStatus.IN_TRANSIT == bus_state:
                time_of_arrival = state.buses[bus_id].t_state_change
                current_block_trip = state.buses[bus_id].current_block_trip
                bus_block_trips = state.buses[bus_id].bus_block_trips
                current_stop_number = state.buses[bus_id].current_stop_number
                current_stop_id = self.get_stop_id_at_number(current_block_trip, current_stop_number)
                last_stop_number = self.get_last_stop_number_on_trip(current_block_trip)

                state.buses[bus_id].current_stop = current_stop_id

                # If valid stop
                if current_stop_number >= 0:
                    res = self.pickup_passengers(time_of_arrival, bus_id, current_stop_id, state)

                # No next stop but maybe has next trips? (will check in idle_update)
                if current_stop_number == last_stop_number:
                    state.buses[bus_id].current_stop_number = 0
                    state.buses[bus_id].status = BusStatus.IDLE

                # Going to next stop
                else:
                    state.buses[bus_id].current_stop_number = current_stop_number + 1
                    travel_time = 100
                    time_to_state_change = time_of_arrival + dt.timedelta(seconds=travel_time)
                    state.buses[bus_id].t_state_change = time_to_state_change
                    event = Event(event_type=EventType.VEHICLE_ARRIVE_AT_STOP,
                                  time=time_to_state_change,
                                  type_specific_information={'bus_id': bus_id,
                                                             'current_block_trip': current_block_trip,
                                                             'stop': state.buses[bus_id].current_stop_number})
                    new_events.append(event)

        # print(f"Event: {event}")
        return new_events

    def pickup_passengers(self, _new_time, bus_id, stop_id, full_state):
        bus_object = full_state.buses[bus_id]
        stop_object = full_state.stops[stop_id]

        vehicle_capacity = bus_object.capacity
        current_block_trip = bus_object.current_block_trip
        current_stop_number = bus_object.current_stop_number
        current_load = bus_object.current_load

        passenger_waiting = stop_object.passenger_waiting
        passenger_arrival_time = _new_time

        route_id_dir = self.get_route_id_dir_for_trip(current_block_trip)
        last_stop_in_trip = self.get_last_stop_number_on_trip(current_block_trip)

        ons = 0
        offs = 0

        if not passenger_waiting:
            return True

        # TODO: Pickup passengers and retain only the latest arrivals
        picked_up_list = []
        if route_id_dir in passenger_waiting:
            for passenger_arrival_time, sampled_data in passenger_waiting[route_id_dir].items():
                # assert passenger_arrival_time <= _new_time
                remaining = sampled_data['remaining']
                sampled_ons = sampled_data['ons']
                sampled_offs = sampled_data['offs']

                if remaining > 0:
                    sampled_ons = remaining

                ons += sampled_ons
                offs += sampled_offs

                picked_up_list.append(passenger_arrival_time)

        if offs > bus_object.current_load:
            offs = bus_object.current_load

        if (bus_object.current_load + ons - offs) > vehicle_capacity:
            remaining = bus_object.current_load + ons - offs - vehicle_capacity
            got_on_bus = max(0, ons - remaining)
        else:
            got_on_bus = ons
            remaining = 0

        # Special cases for the first and last stops
        if current_stop_number == 0:
            offs = 0
        elif current_stop_number == last_stop_in_trip:
            offs = bus_object.current_load
            got_on_bus = 0
            remaining = 0

        # Delete passenger_waiting
        if remaining == 0:
            passenger_waiting[route_id_dir] = {}
        else:
            passenger_waiting[route_id_dir] = {
                passenger_arrival_time: {'got_on_bus': got_on_bus, 'remaining': remaining,
                                         'block_trip': current_block_trip,
                                         'ons': ons, 'offs': offs}}

        stop_object.passenger_waiting[route_id_dir] = passenger_waiting[route_id_dir]
        stop_object.total_passenger_ons += got_on_bus
        stop_object.total_passenger_offs += offs

        bus_object.current_load = bus_object.current_load + got_on_bus - offs
        bus_object.total_passengers_served += got_on_bus
        bus_object.total_stops += 1

        return True

    def get_last_stop_number_on_trip(self, current_block_trip):
        current_trip = current_block_trip[1]
        trip_data = self.trip_plan[current_trip]
        return trip_data['last_stop_sequence']

    def get_stop_id_at_number(self, current_block_trip, current_stop_sequence):
        if current_stop_sequence == -1:
            return None
        current_trip = current_block_trip[1]
        trip_data = self.trip_plan[current_trip]
        return trip_data['stop_id_original'][current_stop_sequence]

    def get_route_id_dir_for_trip(self, current_block_trip):
        current_trip = current_block_trip[1]
        trip_data = self.trip_plan[current_trip]
        route_id = trip_data['route_id']
        route_direction = trip_data['route_direction']
        return str(route_id) + "_" + route_direction
