from Environment.enums import BusStatus, EventType, ActionType, LogType, BusType
from Environment.DataStructures.Event import Event
from src.utils import *
import datetime as dt
import copy


class EnvironmentModelFast:

    def __init__(self, travel_model, logger):
        self.travel_model = travel_model
        self.logger = logger

        self.served_trips = []
        self.served_buses = []

    def update(self, state, curr_event, passenger_arrival_distribution):
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
            log(self.logger, new_time, f"Bus {event_bus_id} broken down before stop {current_stop}", LogType.ERROR)

        elif curr_event.event_type == EventType.VEHICLE_START_TRIP:
            additional_info = curr_event.type_specific_information
            bus_id = additional_info['bus_id']
            bus_state = state.buses[bus_id].status
            bus_type = state.buses[bus_id].type

            if BusStatus.IDLE == bus_state:
                if len(state.buses[bus_id].bus_block_trips) > 0:
                    time_of_activation = new_time
                    # time_of_activation = state.buses[bus_id].t_state_change

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

            # # TODO: Check if another event supercedes this one?
            elif BusStatus.ALLOCATION == bus_state:
                print("YES")
                pass
            #     ofb_obj = state.buses[bus_id]
            #     current_stop = additional_info['current_stop']
            #     reallocation_stop = additional_info['reallocation_stop']
            #     time_added = additional_info['time_added']
            #     # time_of_activation = state.buses[bus_id].t_state_change
            #     time_of_activation = new_time
            #
            #     travel_time = self.travel_model.get_travel_time_from_stop_to_stop(current_stop,
            #                                                                       reallocation_stop,
            #                                                                       state.time)
            #
            #     distance_to_next_stop = self.travel_model.get_distance_from_stop_to_stop(current_stop,
            #                                                                              reallocation_stop,
            #                                                                              state.time)
            #     time_to_state_change = time_of_activation + dt.timedelta(seconds=travel_time)
            #     ofb_obj.t_state_change = time_to_state_change
            #     ofb_obj.distance_to_next_stop = distance_to_next_stop
            #
            #     event = Event(event_type=EventType.VEHICLE_ARRIVE_AT_STOP,
            #                   time=time_to_state_change,
            #                   type_specific_information={'bus_id': bus_id})
            #     new_events.append(event)

        elif curr_event.event_type == EventType.PASSENGER_LEAVE_STOP:
            pass

        elif curr_event.event_type == EventType.VEHICLE_ARRIVE_AT_STOP:
            additional_info = curr_event.type_specific_information
            bus_id = additional_info['bus_id']
            bus_state = state.buses[bus_id].status

            action = additional_info.get("action")

            # HACK
            if action == ActionType.OVERLOAD_ALLOCATE:
                state.buses[bus_id].status = BusStatus.ALLOCATION
                bus_state = BusStatus.ALLOCATION

            elif action == ActionType.OVERLOAD_TO_BROKEN or action == ActionType.OVERLOAD_DISPATCH:
                state.buses[bus_id].status = BusStatus.IN_TRANSIT
                bus_state = BusStatus.IN_TRANSIT
            # END HACK

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
                    state.buses[bus_id].total_service_time += (new_time - state.buses[bus_id].time_at_last_stop).total_seconds()
                state.buses[bus_id].time_at_last_stop = new_time
                state.buses[bus_id].current_stop = current_stop_id

                if current_stop_number >= 0:
                    self.handle_bus_arrival(time_of_arrival, bus_id, state, passenger_arrival_distribution)
                    res = self.pickup_passengers(time_of_arrival, bus_id, current_stop_id, state)

                # No next stop but maybe has next trips? (will check in idle_update)
                if current_stop_number == last_stop_number:
                    state.buses[bus_id].current_stop_number = 0
                    state.buses[bus_id].status = BusStatus.IDLE
                    state.buses[bus_id].t_state_change = new_time
                    # HACK: Again with the travel time
                    time_of_arrival = max(new_time, time_of_arrival)
                    curr_event = Event(event_type=EventType.VEHICLE_START_TRIP,
                                       time=time_of_arrival,
                                       type_specific_information={'bus_id': bus_id,
                                                                  'extra': 'EndOfTrip'})
                    new_events.append(curr_event)

                # Going to next stop
                else:
                    state.buses[bus_id].current_stop_number = current_stop_number + 1
                    travel_time, distance = self.travel_model.get_traveltime_distance_from_depot(current_block_trip,
                                                                                                 current_stop_id,
                                                                                                 state.buses[
                                                                                                     bus_id].current_stop_number)
                    scheduled_arrival_time = self.travel_model.get_scheduled_arrival_time(current_block_trip,
                                                                                          state.buses[
                                                                                              bus_id].current_stop_number)
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

                    # HACK: This shouldn't happen (where a new event is earlier than the current time)
                    time_to_state_change = max(time_to_state_change, new_time)

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
                state.buses[bus_id].time_at_last_stop = new_time
                log(self.logger, new_time, f"Reallocated Bus {bus_id} to {state.buses[bus_id].current_stop}")
                log(self.logger, new_time, f"Bus {bus_id} deadkms: {state.buses[bus_id].total_deadkms_moved:.2f}")

            elif BusStatus.BROKEN == bus_state:
                pass

        # self.cleanup_stops(new_time, state)
        state.time = new_time
        return new_events

    # Issue: it duplicates the values of walk aways
    # since this gets passed multiple times, douvle check delete
    # Issue 2: How to know if its been 30 min. if only running at the end.
    def cleanup_stops(self, _new_time, full_state):
        for_deletion = []
        for stop_id, stop_object in full_state.stops.items():
            passenger_waiting = stop_object.passenger_waiting
            if passenger_waiting is None:
                continue
            for route_id_dir, stop_route_data in passenger_waiting.items():
                for passenger_arrival_time, sampled_data in stop_route_data.items():
                    if _new_time - passenger_arrival_time >= dt.timedelta(minutes=PASSENGER_TIME_TO_LEAVE):
                        sampled_ons = 0
                        sampled_offs = 0
                        remaining = sampled_data.get("remaining", 0)
                        ons = sampled_data.get("ons", 0)
                        got_on_bus = sampled_data.get("got_on_bus", 0)
                        if got_on_bus == 0:
                            walk_aways = ons
                        else:
                            walk_aways = remaining
                        stop_object.total_passenger_walk_away += walk_aways
                        for_deletion.append((stop_id, route_id_dir, passenger_arrival_time))

        for deletion in for_deletion:
            stop_id = deletion[0]
            route_id_dir = deletion[1]
            time_key = deletion[2]
            del full_state.stops[stop_id].passenger_waiting[route_id_dir][time_key]
        return True

    def handle_re_reallocation(self):
        """
        Handles any RE-ASSIGNMENT of a bus currently being reallocated.
        Includes being assigned as DISPATCH, ALLOCATION or to cover BROKEN buses.
        1. Removes any other events in the future for the current overflow bus.
        2. Finds/Interpolates location of current overflow bus.
        """


        pass

    def handle_bus_arrival(self, _new_time, bus_id, full_state, passenger_arrival_distribution):
        current_block_trip = full_state.buses[bus_id].current_block_trip
        bus_block_trips = full_state.buses[bus_id].bus_block_trips
        current_stop_number = full_state.buses[bus_id].current_stop_number
        current_stop_id = self.travel_model.get_stop_id_at_number(current_block_trip, current_stop_number)
        last_stop_number = self.travel_model.get_last_stop_number_on_trip(current_block_trip)
        route_id_dir = self.travel_model.get_route_id_dir_for_trip(current_block_trip)

        # key = (route_id_dir,block,stop_sequence,stop_id_original,pd.Timestamp(scheduled_time[stop_sequence])]
        block_abbr = int(current_block_trip[0])
        scheduled_arrival_time = self.travel_model.get_scheduled_arrival_time(current_block_trip,
                                                                              current_stop_number)

        key = (route_id_dir, block_abbr, current_stop_number + 1, current_stop_id, scheduled_arrival_time)
        val = passenger_arrival_distribution[key]

        curr_stop_loads = val['sampled_loads']
        curr_stop_ons = val['ons']
        curr_stop_offs = val['offs']

        stop_object = full_state.stops[current_stop_id]
        passenger_waiting = stop_object.passenger_waiting
        if passenger_waiting is None:
            # scheduled_arrival_time = scheduled_arrival_time - dt.timedelta(minutes=EARLY_PASSENGER_DELTA_MIN)
            passenger_waiting = {}
            passenger_waiting[route_id_dir] = {}
            passenger_waiting[route_id_dir][scheduled_arrival_time] = {'got_on_bus': 0,
                                                                       'remaining': 0,
                                                                       'block_trip': "",
                                                                       'ons': curr_stop_ons,
                                                                       'offs': curr_stop_offs}

        stop_object.passenger_waiting = passenger_waiting

    def pickup_passengers(self, _new_time, bus_id, stop_id, full_state):
        bus_object = full_state.buses[bus_id]
        stop_object = full_state.stops[stop_id]

        vehicle_capacity = bus_object.capacity
        current_block_trip = bus_object.current_block_trip
        current_stop_number = bus_object.current_stop_number
        current_load = bus_object.current_load
        bus_arrival_time = _new_time

        passenger_waiting = stop_object.passenger_waiting
        # passenger_arrival_time = _new_time

        route_id_dir = self.travel_model.get_route_id_dir_for_trip(current_block_trip)
        last_stop_in_trip = self.travel_model.get_last_stop_number_on_trip(current_block_trip)
        scheduled_arrival_time = self.travel_model.get_scheduled_arrival_time(current_block_trip,
                                                                              current_stop_number)

        ons = 0
        offs = 0

        if not passenger_waiting:
            return True

        picked_up_list = []
        for_deletion = []
        if route_id_dir in passenger_waiting:
            for passenger_arrival_time, sampled_data in passenger_waiting[route_id_dir].items():
                assert passenger_arrival_time <= bus_arrival_time
                sampled_ons = 0
                sampled_offs = 0
                remaining = 0
                if bus_arrival_time - passenger_arrival_time <= dt.timedelta(minutes=PASSENGER_TIME_TO_LEAVE):
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

                # Substitute for the leaving events
                elif passenger_arrival_time < (bus_arrival_time - dt.timedelta(minutes=PASSENGER_TIME_TO_LEAVE)):
                    sampled_ons = 0
                    sampled_offs = 0
                    remaining = sampled_data.get("remaining", 0)
                    ons = sampled_data.get("ons", 0)

                    got_on_bus = sampled_data.get("got_on_bus", 0)
                    if got_on_bus == 0:
                        walk_aways = ons
                    else:
                        walk_aways = remaining
                    stop_object.total_passenger_walk_away += walk_aways
                    for_deletion.append((route_id_dir, passenger_arrival_time))
                    ons = 0
                    offs += sampled_offs
                    got_on_bus = 0
                    if remaining > 0:
                        log(self.logger, _new_time, f"{remaining} people left stop {stop_id}", LogType.ERROR)
                    remaining = 0

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

        # TODO: Adjust passenger arrival time to the latest passenger?
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

        for deletion in for_deletion:
            route_id_dir = deletion[0]
            time_key = deletion[1]
            if route_id_dir in stop_object.passenger_waiting:
                if time_key in stop_object.passenger_waiting[route_id_dir]:
                    del stop_object.passenger_waiting[route_id_dir][time_key]

        return True

    def take_action(self, state, action):
        # print("take_action")
        action_type = action['type']
        ofb_id = action['overload_bus']
        # ('HILLOMNF', '7_TO DOWNTOWN', datetime.datetime(2021, 8, 23, 14, 13, 11), 8.0, ('5692', '246343'))

        new_events = []

        if ActionType.OVERLOAD_DISPATCH == action_type:
            ofb_obj = state.buses[ofb_id]

            action_info = action["info"]
            stop_id = action_info[0]
            route_id_dir = action_info[1]
            arrival_time = action_info[2]
            remaining = action_info[3]
            current_block_trip = action_info[4]

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
                          type_specific_information={'bus_id': ofb_id,
                                                     'action': ActionType.OVERLOAD_DISPATCH})

            new_events.append(event)

            log(self.logger, state.time,
                f"Dispatching overflow bus: {ofb_id} from {ofb_obj.current_stop} @ stop {stop_id}",
                LogType.ERROR)

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
            ofb_obj.total_passengers_served += ofb_obj.current_load

            # Deactivate broken_bus_obj
            # broken_bus_obj.total_passengers_served = broken_bus_obj.total_passengers_served - broken_bus_obj.current_load
            broken_bus_obj.current_load = 0
            broken_bus_obj.current_block_trip = None
            broken_bus_obj.bus_block_trips = []
            broken_bus_obj.total_passengers_served -= ofb_obj.current_load

            event = Event(event_type=EventType.VEHICLE_START_TRIP,
                          time=state.time,
                          type_specific_information={'bus_id': ofb_id,
                                                     'action': ActionType.OVERLOAD_TO_BROKEN})
            new_events.append(event)

            log(self.logger, state.time,
                f"Sending takeover overflow bus: {ofb_id} from {ofb_obj.current_stop} @ stop {broken_bus_obj.current_stop}",
                LogType.ERROR)

        elif ActionType.OVERLOAD_ALLOCATE == action_type:
            ofb_obj = state.buses[ofb_id]
            current_stop = ofb_obj.current_stop
            action_info = action["info"]
            reallocation_stop = action_info

            travel_time = self.travel_model.get_travel_time_from_stop_to_stop(current_stop, reallocation_stop,
                                                                              state.time)
            distance_to_next_stop = self.travel_model.get_distance_from_stop_to_stop(current_stop, reallocation_stop,
                                                                                     state.time)

            time_to_state_change = state.time + dt.timedelta(seconds=travel_time)
            ofb_obj.current_stop = reallocation_stop
            ofb_obj.t_state_change = time_to_state_change
            # ofb_obj.time_at_last_stop = state.time
            ofb_obj.distance_to_next_stop = distance_to_next_stop
            ofb_obj.status = BusStatus.ALLOCATION

            event = Event(event_type=EventType.VEHICLE_ARRIVE_AT_STOP,
                          time=time_to_state_change,
                          type_specific_information={'bus_id': ofb_id,
                                                     'current_stop': current_stop,
                                                     'reallocation_stop': reallocation_stop,
                                                     'action': ActionType.OVERLOAD_ALLOCATE,
                                                     'time_added': state.time})

            # _events = [be for be in state.bus_events if (be.event_type == EventType.VEHICLE_ARRIVE_AT_STOP) ]
            new_events.append(event)
            # new_events = self.dispatch_policy.
            log(self.logger, state.time,
                f"Reallocating overflow bus: {ofb_id} from {current_stop} to {reallocation_stop}", LogType.INFO)

        elif ActionType.NO_ACTION == action_type:
            # Do nothing
            pass

        return new_events, state.time
