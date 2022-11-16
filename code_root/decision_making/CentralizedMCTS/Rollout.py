import copy
import time
import random
import datetime as dt
from Environment.DataStructures.State import State
from decision_making.CentralizedMCTS.DataStructures.TreeNode import TreeNode
from Environment.enums import EventType, ActionType, BusType, BusStatus
from src.utils import *
import itertools

class BareMinimumRollout:
    """
    Bare minimum rollout, send the nearest bus (if available) to cover for a broken down bus.
    """

    def __init__(self):
        self.deep_copy_time = 0
        self.rollout_horizon_delta_t = 60 * 60 * 0.6  # 60*60*N for N hour horizon (0.6)
        # self.rollout_horizon_delta_t = None
        
        self.horizon_time_limit = None
            
        self.total_walkaways = 0

    def rollout(self,
                node,
                environment_model,
                discount_factor,
                solve_start_time,
                passenger_arrival_distribution):
        s_copy_time = time.time()
        self.debug_rewards = []
        self.total_walkaways = 0
        self.passenger_arrival_distribution = passenger_arrival_distribution

        truncated_events = copy.copy(node.future_events_queue)
        
        if self.rollout_horizon_delta_t is not None:
            self.horizon_time_limit = solve_start_time + dt.timedelta(seconds=self.rollout_horizon_delta_t)
            truncated_events = [event for event in truncated_events if
                                solve_start_time <= event.time <= self.horizon_time_limit]
        else:
            self.horizon_time_limit = solve_start_time + dt.timedelta(days=1)
        
        _state = State(
            stops=copy.deepcopy(node.state.stops),
            buses=copy.deepcopy(node.state.buses),
            bus_events=None,
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

        return _node.reward_to_here

    """
    SendNearestDispatch if a vehicle is broken, else do nothing.
    However, breakdown events are not generated in the event chains.
    """
    def rollout_iter(self, node, environment_model, discount_factor, solve_start_time):
        # rollout1
        # valid_actions, _ = environment_model.generate_possible_actions(node.state,
        #                                                                node.event_at_node,
        #                                                                action_type=ActionType.OVERLOAD_ALL)
        # random.seed(100)
        # action_to_take = random.choice(valid_actions)
        
        # action_to_take = environment_model.get_rollout_actions(node.state, valid_actions)
        
        # rollout2
        if node.event_at_node.event_type == EventType.VEHICLE_BREAKDOWN:
            idle_overload_buses = []
            for bus_id, bus_obj in node.state.buses.items():
                if (bus_obj.type == BusType.OVERLOAD) and \
                        ((bus_obj.status == BusStatus.IDLE) or (bus_obj.status == BusStatus.ALLOCATION)):
                    if len(bus_obj.bus_block_trips) <= 0:
                        idle_overload_buses.append(bus_id)

            broken_buses = []
            for bus_id, bus_obj in node.state.buses.items():
                if bus_obj.status == BusStatus.BROKEN:
                    if bus_obj.current_block_trip is not None:
                        broken_buses.append(bus_id)
            if len(idle_overload_buses) > 0:
                valid_actions = [[ActionType.OVERLOAD_TO_BROKEN], idle_overload_buses, broken_buses]
                valid_actions = list(itertools.product(*valid_actions))
                random.seed(100)
                action_to_take = random.choice(valid_actions)
                action_to_take = {'type': action_to_take[0], 'overload_bus': action_to_take[1], 'info': action_to_take[2]}
            else:
                action_to_take = {'type': ActionType.NO_ACTION, 'overload_bus': None, 'info': None}
        else:
            action_to_take = {'type': ActionType.NO_ACTION, 'overload_bus': None, 'info': None}

        immediate_reward, new_events, event_time = environment_model.take_action(node.state, action_to_take)
        
        # Limit the number of new events generated based on the time horizon!!!
        if len(new_events) > 0:
            # TODO: Check if some are still added here...
            if new_events[0].time <= self.horizon_time_limit:
                node.future_events_queue.append(new_events[0])
                node.future_events_queue.sort(key=lambda _: _.time)

        node.depth += 1
        node.event_at_node = node.future_events_queue.pop(0)
        new_events = self.process_event(node.state, node.event_at_node, environment_model)
        if len(new_events) > 0:
            # TODO: Check if some are still added here...
            if new_events[0].time <= self.horizon_time_limit:
                node.future_events_queue.append(new_events[0])
                node.future_events_queue.sort(key=lambda _: _.time)

        discounted_immediate_score = self.standard_discounted_score(immediate_reward,
                                                                    event_time - solve_start_time,
                                                                    discount_factor)

        node.reward_to_here = node.reward_to_here + discounted_immediate_score
        # print(len(node.future_events_queue))

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
        new_events = environment_model.update(state, event, self.passenger_arrival_distribution)
        return new_events
