from Environment.enums import BusStatus, EventType, ActionType, LogType, BusType
from Environment.DataStructures.Event import Event
from src.utils import *
import datetime as dt
import pandas as pd
import copy


class EnvironmentModelFast:

    def __init__(self, travel_model, logger):
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
        log(self.logger, new_time, f"Event: {curr_event}", LogType.INFO)

        new_events = []
        if curr_event.event_type == EventType.VEHICLE_BREAKDOWN:
            type_specific_information = curr_event.type_specific_information
            event_bus_id = type_specific_information['bus_id']
            current_block_trip = state.buses[event_bus_id].current_block_trip
            state.buses[event_bus_id].status = BusStatus.BROKEN
            current_stop = state.buses[event_bus_id].current_stop
            log(self.logger, new_time, f"Bus {event_bus_id} broke down before stop {current_stop}", LogType.ERROR)

        elif curr_event.event_type == EventType.PASSENGER_ARRIVE_STOP:
            additional_info = curr_event.type_specific_information
            arrival_time = curr_event.time
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

        elif curr_event.event_type == EventType.PASSENGER_LEAVE_STOP:
            additional_info = curr_event.type_specific_information
            curr_route_id_dir = additional_info['route_id_dir']
            curr_stop_id = additional_info['stop_id']
            time_key = additional_info['time']
            passenger_waiting = state.stops[curr_stop_id].passenger_waiting

            # HACK: Not sure if this is correct
            if passenger_waiting is None:
                return []

            if curr_route_id_dir not in passenger_waiting:
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

        elif curr_event.event_type == EventType.VEHICLE_START_TRIP:
            additional_info = curr_event.type_specific_information
            bus_id = additional_info['bus_id']
            bus_state = state.buses[bus_id].status
            bus_type = state.buses[bus_id].type

            if BusStatus.IDLE == bus_state:
                if len(state.buses[bus_id].bus_block_trips) > 0:
                    time_of_activation = state.buses[bus_id].t_state_change

                    state.buses[bus_id].status = BusStatus.IN_TRANSIT
                    current_block_trip = state.buses[bus_id].bus_block_trips.pop(0)
                    state.buses[bus_id].current_block_trip = current_block_trip
                    current_depot = state.buses[bus_id].current_stop
                    current_stop_number = state.buses[bus_id].current_stop_number
                    scheduled_arrival_time = self.travel_model.get_scheduled_arrival_time(current_block_trip,
                                                                                          current_stop_number)

                    travel_time, distance = self.travel_model.get_traveltime_distance_from_depot(current_block_trip,
                                                                                                 current_depot,
                                                                                                 current_stop_number)
                    if BusType.OVERLOAD == bus_type:
                        state.buses[bus_id].total_deadkms_moved += distance
                    else:
                        state.buses[bus_id].distance_to_next_stop = distance

                    time_to_state_change = time_of_activation + dt.timedelta(seconds=travel_time)
                    # Buses should start either at the scheduled time, or if they are late, should start as soon as possible.
                    time_to_state_change = max(time_to_state_change, scheduled_arrival_time)
                    state.buses[bus_id].t_state_change = time_to_state_change
                    state.buses[bus_id].time_at_last_stop = time_of_activation

                    curr_event = Event(event_type=EventType.VEHICLE_ARRIVE_AT_STOP,
                                       time=time_to_state_change,
                                       type_specific_information={'bus_id': bus_id,
                                                                  'current_block_trip': current_block_trip,
                                                                  'stop': state.buses[bus_id].current_stop_number})
                    new_events.append(curr_event)

                else:
                    # no more trips left
                    pass

            elif BusStatus.IN_TRANSIT == bus_state:
                pass

            elif BusStatus.BROKEN == bus_state:
                pass

            elif BusStatus.ALLOCATION == bus_state:
                ofb_obj = state.buses[bus_id]
                current_stop = additional_info['current_stop']
                reallocation_stop = additional_info['reallocation_stop']

                travel_time = self.travel_model.get_travel_time_from_stop_to_stop(current_stop,
                                                                                  reallocation_stop,
                                                                                  state.time)

                distance_to_next_stop = self.travel_model.get_distance_from_stop_to_stop(current_stop,
                                                                                         reallocation_stop,
                                                                                         state.time)
                time_to_state_change = state.time + dt.timedelta(seconds=travel_time)
                ofb_obj.t_state_change = time_to_state_change
                ofb_obj.status = BusStatus.ALLOCATION
                ofb_obj.time_at_last_stop = state.time
                ofb_obj.distance_to_next_stop = distance_to_next_stop

                event = Event(event_type=EventType.VEHICLE_ARRIVE_AT_STOP,
                              time=time_to_state_change,
                              type_specific_information={'bus_id': bus_id})
                new_events.append(event)

        elif curr_event.event_type == EventType.VEHICLE_ARRIVE_AT_STOP:
            additional_info = curr_event.type_specific_information
            bus_id = additional_info['bus_id']
            bus_state = state.buses[bus_id].status

            if BusStatus.IDLE == bus_state:
                # raise "Should not have an IDLE bus arriving at a stop."
                pass

            elif BusStatus.IN_TRANSIT == bus_state:
                time_of_arrival = state.buses[bus_id].t_state_change

                current_block_trip = state.buses[bus_id].current_block_trip
                bus_block_trips = state.buses[bus_id].bus_block_trips
                current_stop_number = state.buses[bus_id].current_stop_number
                current_stop_id = self.travel_model.get_stop_id_at_number(current_block_trip, current_stop_number)
                last_stop_number = self.travel_model.get_last_stop_number_on_trip(current_block_trip)

                # Bus running time
                if state.buses[bus_id].time_at_last_stop:
                    state.buses[bus_id].total_service_time += (
                                new_time - state.buses[bus_id].time_at_last_stop).total_seconds()
                state.buses[bus_id].time_at_last_stop = time_of_arrival
                state.buses[bus_id].current_stop = current_stop_id

                # If valid stop
                if current_stop_number >= 0:
                    res = self.pickup_passengers(new_time, bus_id, current_stop_id, state)

                # No next stop but maybe has next trips? (will check in idle_update)
                if current_stop_number == last_stop_number:
                    state.buses[bus_id].current_stop_number = 0
                    state.buses[bus_id].status = BusStatus.IDLE

                # Going to next stop
                else:
                    state.buses[bus_id].current_stop_number = current_stop_number + 1

                    travel_time, distance = self.travel_model.get_traveltime_distance_from_depot(current_block_trip,
                                                                                                 current_stop_id,
                                                                                                 state.buses[bus_id].current_stop_number)

                    scheduled_arrival_time = self.travel_model.get_scheduled_arrival_time(current_block_trip,
                                                                                          state.buses[bus_id].current_stop_number)

                    time_to_state_change = time_of_arrival + dt.timedelta(seconds=travel_time)

                    # Taking into account delay time
                    if scheduled_arrival_time < time_to_state_change:
                        delay_time = time_to_state_change - scheduled_arrival_time
                        state.buses[bus_id].delay_time += delay_time.total_seconds()

                    # TODO: Not the best place to put this, Dwell time
                    elif scheduled_arrival_time > time_to_state_change:
                        dwell_time = scheduled_arrival_time - time_to_state_change
                        state.buses[bus_id].dwell_time += dwell_time.total_seconds()
                        time_to_state_change = time_to_state_change + dwell_time

                    state.buses[bus_id].t_state_change = time_to_state_change

                    # For distance
                    state.buses[bus_id].total_servicekms_moved += state.buses[bus_id].distance_to_next_stop
                    state.buses[bus_id].distance_to_next_stop = distance

                    curr_event = Event(event_type=EventType.VEHICLE_ARRIVE_AT_STOP,
                                       time=time_to_state_change,
                                       type_specific_information={'bus_id': bus_id,
                                                                  'current_block_trip': current_block_trip,
                                                                  'stop': state.buses[bus_id].current_stop_number})
                    new_events.append(curr_event)

            elif BusStatus.ALLOCATION == bus_state:
                distance_to_next_stop = state.buses[bus_id].distance_to_next_stop
                state.buses[bus_id].total_deadkms_moved += distance_to_next_stop
                state.buses[bus_id].status = BusStatus.IDLE
                log(self.logger, new_time, f"Reallocated Bus {bus_id} to {state.buses[bus_id].current_stop}")

            elif BusStatus.BROKEN == bus_state:
                pass

        state.time = new_time
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

        route_id_dir = self.travel_model.get_route_id_dir_for_trip(current_block_trip)
        last_stop_in_trip = self.travel_model.get_last_stop_number_on_trip(current_block_trip)
        scheduled_arrival_time = self.travel_model.get_scheduled_arrival_time(current_block_trip,
                                                                              current_stop_number)

        ons = 0
        offs = 0

        if not passenger_waiting:
            return True

        # TODO: if bus arrives 30 minutes after passenger arrives (consider passengers as walk away)
        picked_up_list = []
        if route_id_dir in passenger_waiting:
            for passenger_arrival_time, sampled_data in passenger_waiting[route_id_dir].items():
                assert passenger_arrival_time <= _new_time
                remaining = sampled_data['remaining']
                sampled_ons = sampled_data['ons']
                sampled_offs = sampled_data['offs']

                # QUESTION: I think this needs to be += and not just = in case ons is non-zero.
                if remaining > 0:
                    # sampled_ons += remaining
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
            log(self.logger, _new_time, f"Bus {bus_id} left {remaining} people at stop {stop_id}", LogType.ERROR)

        stop_object.passenger_waiting[route_id_dir] = passenger_waiting[route_id_dir]
        stop_object.total_passenger_ons += got_on_bus
        stop_object.total_passenger_offs += offs

        bus_object.current_load = bus_object.current_load + got_on_bus - offs
        bus_object.total_passengers_served += got_on_bus
        bus_object.total_stops += 1

        log_str = f"""Bus {bus_id} on trip: {current_block_trip[1]} scheduled for {scheduled_arrival_time} \
arrives at @ {stop_id}: got_on:{got_on_bus:.0f}, on:{ons:.0f}, offs:{offs:.0f}, \
remain:{remaining:.0f}, bus_load:{bus_object.current_load:.0f}"""
        log(self.logger, _new_time, log_str, LogType.INFO)

        return True

    def take_action(self, state, action):
        # print("take_action")
        action_type = action['type']
        ofb_id = action['overload_bus']
        # ('HILLOMNF', '7_TO DOWNTOWN', datetime.datetime(2021, 8, 23, 14, 13, 11), 8.0, ('5692', '246343'))

        new_events = []

        # TODO: Make this better looking
        # Send to stop

        if ActionType.OVERLOAD_DISPATCH == action_type:
            ofb_obj = state.buses[ofb_id]

            action_info = action["info"]
            stop_id = action_info[0]
            route_id_dir = action_info[1]
            arrival_time = action_info[2]
            remaining = action_info[3]
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

            log(self.logger, state.time,
                f"Dispatching overflow bus: {ofb_id} from {ofb_obj.current_stop} @ stop {stop_id}",
                LogType.ERROR)

            # self.served_trips.append(current_block_trip)

        # Take over broken bus
        elif ActionType.OVERLOAD_TO_BROKEN == action_type:
            ofb_obj = state.buses[ofb_id]
            action_info = action["info"]
            broken_bus_id = action_info

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

            log(self.logger, state.time,
                f"Sending takeover overflow bus: {ofb_id} from {ofb_obj.current_stop} @ stop {broken_bus_obj.current_stop}",
                LogType.ERROR)

        elif ActionType.OVERLOAD_ALLOCATE == action_type:
            # print(f"Random Coord: {action}")
            ofb_obj = state.buses[ofb_id]
            current_stop = ofb_obj.current_stop
            action_info = action["info"]
            reallocation_stop = action_info

            travel_time = self.travel_model.get_travel_time_from_stop_to_stop(current_stop, reallocation_stop,
                                                                              state.time)
            distance_to_next_stop = self.travel_model.get_distance_from_stop_to_stop(current_stop, reallocation_stop,
                                                                                     state.time)

            ofb_obj.current_stop = reallocation_stop
            ofb_obj.t_state_change = state.time
            ofb_obj.status = BusStatus.ALLOCATION
            ofb_obj.time_at_last_stop = state.time
            ofb_obj.distance_to_next_stop = distance_to_next_stop

            event = Event(event_type=EventType.VEHICLE_START_TRIP,
                          time=state.time,
                          type_specific_information={'bus_id': ofb_id,
                                                     'current_stop': current_stop,
                                                     'reallocation_stop': reallocation_stop})
            new_events.append(event)
            # new_events = self.dispatch_policy.
            log(self.logger, state.time,
                f"Reallocating overflow bus: {ofb_id} from {current_stop} to {reallocation_stop}", LogType.INFO)

        elif ActionType.NO_ACTION == action_type:
            # Do nothing
            pass

        return new_events, state.time
