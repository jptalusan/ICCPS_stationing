from Environment.enums import BusStatus, BusType, ActionType
import random


class SendNearestDispatchPolicy:
    
    def __init__(self, travel_model) -> None:
        self.travel_model = travel_model
        pass

    # _valid_actions = [[ActionType.OVERLOAD_DISPATCH], idle_overload_buses, stops_with_left_behind_passengers]
    # _valid_actions = [[ActionType.OVERLOAD_TO_BROKEN], idle_overload_buses, broken_buses]

    def select_overload_to_dispatch(self, state, actions):
        if len(actions) <= 0:
            return None

        actions_with_distance = []

        for action in actions:
            action_type = action['type']
            overload_bus = action['overload_bus']
            info = action['info']

            if action_type == ActionType.NO_ACTION:
                return None

            current_stop = state.buses[overload_bus].current_stop
            next_stop = None

            if action_type == ActionType.OVERLOAD_DISPATCH:
                next_stop = info[0]
                # route_id_dir = info[1]
                # passenger_arrival_time = info[2]
                # remaining_count = info[3]
                # block_trip = info[4]
            elif action_type == ActionType.OVERLOAD_TO_BROKEN:
                broken_bus = info
                next_stop = state.buses[info].current_stop
            elif action_type == ActionType.OVERLOAD_ALLOCATE:
                pass
            else:
                raise "Action not supported"

            distance = self.travel_model.get_distance_from_stop_to_stop(current_stop, next_stop, state.time)
            actions_with_distance.append((action, distance))

        actions_with_distance = sorted(actions_with_distance, key=lambda x: x[1], reverse=False)
        # print(actions_with_distance)
        return actions_with_distance[0][0]
