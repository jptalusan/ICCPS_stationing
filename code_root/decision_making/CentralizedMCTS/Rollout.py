import copy
import time
import itertools
from Environment.DataStructures.State import State
from decision_making.CentralizedMCTS.DataStructures.TreeNode import TreeNode
from Environment.enums import EventType, ActionType, BusType, BusStatus


class BareMinimumRollout:
    """
    Bare minimum rollout, send the nearest bus (if available) to cover for a broken down bus.
    """
    def __init__(self):
        self.deep_copy_time = 0

    def rollout(self,
                node,
                environment_model,
                discount_factor,
                solve_start_time):
        s_copy_time = time.time()

        _state = State(
            stops=copy.deepcopy(node.state.stops),
            buses=copy.deepcopy(node.state.buses),
            events=copy.copy(node.future_events_queue),
            # events=copy.copy(node.state.events),
            time=node.state.time,
            active_incidents=[]
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
            future_events_queue=copy.copy(node.future_events_queue)
        )

        self.deep_copy_time += time.time() - s_copy_time

        # Run until all events finish
        while _node.future_events_queue:
            self.rollout_iter(_node, environment_model, discount_factor, solve_start_time)

        return _node.reward_to_here

    """
    SendNearestDispatch if a vehicle is broken, else do nothing.
    """
    def rollout_iter(self, node, environment_model, discount_factor, solve_start_time):
        ## Get valid actions (ValidActions)
        ## Take Action only on breakdowns
        ## Update reward
        _curr_event = node.event_at_node
        immediate_reward = 0
        new_events = None
        event_time = node.state.time
        valid_actions = []
        action_to_take = None
        if _curr_event.event_type == EventType.VEHICLE_BREAKDOWN:
            broken_bus_id = _curr_event.type_specific_information['bus_id']
            action_type = ActionType.OVERLOAD_TO_BROKEN
            idle_overload_buses = []
            for bus_id, bus_obj in node.state.buses.items():
                if (bus_obj.type == BusType.OVERLOAD) and (bus_obj.status == BusStatus.IDLE):
                    idle_overload_buses.append(bus_id)
        
            valid_actions = [[ActionType.OVERLOAD_TO_BROKEN], idle_overload_buses, [broken_bus_id]]
            valid_actions = list(itertools.product(*valid_actions))
            valid_actions = [{'type': _va[0], 'overload_bus': _va[1], 'info': _va[2]} for _va in valid_actions]
            action_to_take = environment_model.get_rollout_actions(node.state, valid_actions)
        
        if not action_to_take:
            action_to_take = {'type': ActionType.NO_ACTION, 'overload_bus': None, 'info': None}
            
        # action_to_take = {'type': ActionType.NO_ACTION, 'overload_bus': None, 'info': None}
        immediate_reward, new_events, event_time = environment_model.take_action(node.state, action_to_take)
        # NOTE: New events returns a list even though its only returning a single event always
        if new_events is not None:
            node.future_events_queue.append(new_events[0])
            node.future_events_queue.sort(key= lambda _: _.time)

        node.depth += 1
        node.event_at_node = node.future_events_queue.pop(0)
        self.process_event(node.state, node.event_at_node, environment_model)

        discounted_immediate_score = self.standard_discounted_score(immediate_reward,
                                                                    event_time - solve_start_time,
                                                                    discount_factor)

        node.reward_to_here = node.reward_to_here + discounted_immediate_score

    def standard_discounted_score(self, reward, time_since_start, discount_factor):
        discount = discount_factor ** time_since_start.total_seconds()
        discounted_reward = discount * reward
        return discounted_reward

    def process_event(self, state, event, environment_model):
        environment_model.update(state, event)
