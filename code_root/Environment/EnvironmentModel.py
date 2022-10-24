from Environment.BusDynamics import BusDynamics
from Environment.StopDynamics import StopDynamics
from Environment.enums import BusStatus, EventType, ActionType, LogType
from Environment.DataStructures.Event import Event
from src.utils import *
import datetime as dt
import copy


class EnvironmentModel:

    def __init__(self, travel_model, logger):
        self.bus_dynamics = BusDynamics(travel_model, logger)
        self.stop_dynamics = StopDynamics(travel_model, logger)
        self.travel_model = travel_model
        self.logger = logger

        self.served_trips = []
        self.served_buses = []

    def update(self, state, curr_event):
        """
        Updates the state to the given time. This is mostly updating the responders
        :param state:
        :param curr_event:
        :return:
        """
        reward = 0
        new_events = []
        new_time = curr_event.time

        try:
            assert new_time >= state.time
        except AssertionError:
            print(curr_event)
            print(new_time)
            print(state.time)
            assert new_time >= state.time

        # print(curr_event)
        log(self.logger, state.time, f"Event: {curr_event}", LogType.INFO)

        if (curr_event.event_type == EventType.PASSENGER_ARRIVE_STOP) or \
           (curr_event.event_type == EventType.PASSENGER_LEAVE_STOP):
            # update the state of a SINGLE stop
            _new_events = self.stop_dynamics.update_stop(curr_event,
                                                         _new_time=new_time,
                                                         full_state=state)
            new_events.extend(_new_events)

        # update the state of EACH bus
        for bus_id, bus_obj in state.buses.items():
            _new_events = self.bus_dynamics.update_bus(curr_event,
                                                       bus_id=bus_id,
                                                       _new_time=new_time,
                                                       full_state=state)
            new_events.extend(_new_events)

        state.time = new_time
        return new_events

    def take_action(self, state, action):
        # print("take_action")
        action_type = action['type']
        ofb_id      = action['overload_bus']
        # ('HILLOMNF', '7_TO DOWNTOWN', datetime.datetime(2021, 8, 23, 14, 13, 11), 8.0, ('5692', '246343'))

        new_events = []

        # TODO: Make this better looking
        # Send to stop

        if ActionType.OVERLOAD_DISPATCH == action_type:
            ofb_obj = state.buses[ofb_id]

            action_info        = action["info"]
            stop_id            = action_info[0]
            route_id_dir       = action_info[1]
            arrival_time       = action_info[2]
            remaining          = action_info[3]
            current_block_trip = action_info[4]

            # if current_block_trip in self.served_trips:
            #     return []

            log(self.logger,
                state.time,
                f"ActionTaken: {action}, .",
                LogType.INFO)
            stop_no = self.travel_model.get_stop_number_at_id(current_block_trip, stop_id)

            ofb_obj.bus_block_trips = [current_block_trip]
            # Because at this point we already set the state to the next stop.
            ofb_obj.current_stop_number = stop_no
            ofb_obj.t_state_change = state.time

            event = Event(event_type=EventType.VEHICLE_START_TRIP,
                          time=state.time,
                          type_specific_information={'bus_id': ofb_id})

            new_events.append(event)

            # self.served_trips.append(current_block_trip)

        # Take over broken bus
        elif ActionType.OVERLOAD_TO_BROKEN == action_type:
            ofb_obj       = state.buses[ofb_id]
            action_info   = action["info"]
            broken_bus_id = action_info

            # if broken_bus_id in self.served_buses:
            #     return []

            log(self.logger, state.time, f"Real world Env Taking action: {action}, .", LogType.INFO)
            broken_bus_obj = state.buses[broken_bus_id]

            current_block_trip = broken_bus_obj.current_block_trip
            stop_no = broken_bus_obj.current_stop_number

            ofb_obj.bus_block_trips = copy.copy([broken_bus_obj.current_block_trip] + broken_bus_obj.bus_block_trips)
            # Remove None, in case bus has not started trip.
            ofb_obj.bus_block_trips = [x for x in ofb_obj.bus_block_trips if x is not None]

            ofb_obj.current_block_trip = None
            # In case bus has not yet started trip.
            if stop_no == 0:
                ofb_obj.current_stop_number = 0
            # Because at this point we already set the state to the next stop.
            else:
                ofb_obj.current_stop_number = stop_no - 1
                
            ofb_obj.t_state_change = state.time

            # Switch passengers
            ofb_obj.current_load = copy.copy(broken_bus_obj.current_load)
            ofb_obj.total_passengers_served = ofb_obj.current_load

            # Deactivate broken_bus_obj
            # broken_bus_obj.total_passengers_served = broken_bus_obj.total_passengers_served - broken_bus_obj.current_load
            broken_bus_obj.current_load = 0
            broken_bus_obj.current_block_trip = None
            broken_bus_obj.bus_block_trips = []

            event = Event(event_type=EventType.VEHICLE_START_TRIP,
                          time=state.time,
                          type_specific_information={'bus_id': ofb_id})
            new_events.append(event)

            # self.served_buses.append(broken_bus_id)
            log(self.logger, state.time, f"Sending takeover overflow bus: {ofb_id} from {ofb_obj.current_stop} @ stop {broken_bus_obj.current_stop}", LogType.ERROR)
            # log(self.logger, state.time, f"Sending takeover overflow bus: {ofb_id} to {broken_bus_obj.current_block_trip} @ stop {broken_bus_obj.current_stop}", LogType.ERROR)

        elif ActionType.OVERLOAD_ALLOCATE == action_type:
            # print(f"Random Coord: {action}")
            ofb_obj = state.buses[ofb_id]
            current_stop = ofb_obj.current_stop
            action_info = action["info"]
            reallocation_stop = action_info

            travel_time = self.travel_model.get_travel_time_from_stop_to_stop(current_stop, reallocation_stop, state.time)
            distance_to_next_stop  = self.travel_model.get_distance_from_stop_to_stop(current_stop, reallocation_stop, state.time)

            ofb_obj.current_stop = reallocation_stop
            ofb_obj.t_state_change = state.time + dt.timedelta(seconds=travel_time)
            ofb_obj.status  = BusStatus.ALLOCATION
            ofb_obj.time_at_last_stop = state.time
            ofb_obj.distance_to_next_stop = distance_to_next_stop

            event = Event(event_type=EventType.VEHICLE_START_TRIP,
                          time=state.time,
                          type_specific_information={'bus_id': ofb_id})
            new_events.append(event)
            # new_events = self.dispatch_policy.
            log(self.logger, state.time, f"Reallocating overflow bus: {ofb_id} from {current_stop} to {reallocation_stop}", LogType.INFO)

        elif ActionType.NO_ACTION == action_type:
            # Do nothing
            pass

        return new_events, state.time
