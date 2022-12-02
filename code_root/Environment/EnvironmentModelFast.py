from Environment.enums import BusStatus, EventType, ActionType, LogType, BusType
from Environment.DataStructures.Event import Event
from src.utils import *
import datetime as dt
import copy
import pandas as pd


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
        :param passenger_arrival_distribution:
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
        log(self.logger, new_time, f"Event: {curr_event}", LogType.DEBUG)

        new_events = []
        if curr_event.event_type == EventType.VEHICLE_BREAKDOWN:
            type_specific_information = curr_event.type_specific_information
            event_bus_id = type_specific_information['bus_id']
            current_block_trip = state.buses[event_bus_id].current_block_trip
            state.buses[event_bus_id].status = BusStatus.BROKEN
            current_stop = state.buses[event_bus_id].current_stop
            log(self.logger, new_time, f"Bus {event_bus_id} broken down before stop {current_stop}", LogType.ERROR)

        elif curr_event.event_type == EventType.VEHICLE_ARRIVE_AT_STOP:
            additional_info = curr_event.type_specific_information
            bus_id = additional_info['bus_id']
            bus_state = state.buses[bus_id].status
            bus_type = state.buses[bus_id].type

            if BusStatus.IDLE == bus_state:
                # raise "Should not have an IDLE bus arriving at a stop."
                pass

            elif BusStatus.IN_TRANSIT == bus_state:
                time_of_arrival = state.buses[bus_id].t_state_change
                current_block_trip = state.buses[bus_id].current_block_trip
                bus_block_trips = state.buses[bus_id].bus_block_trips
                current_stop_number = state.buses[bus_id].current_stop_number

                if current_block_trip is None:
                    return new_events
                current_stop_id = self.travel_model.get_stop_id_at_number(current_block_trip, current_stop_number)
                last_stop_number = self.travel_model.get_last_stop_number_on_trip(current_block_trip)

                # Bus running time
                if state.buses[bus_id].time_at_last_stop:
                    state.buses[bus_id].total_service_time += (
                            new_time - state.buses[bus_id].time_at_last_stop).total_seconds()
                state.buses[bus_id].time_at_last_stop = new_time
                state.buses[bus_id].current_stop = current_stop_id

                if current_stop_number >= 0:
                    self.handle_bus_arrival(time_of_arrival, bus_id, state, passenger_arrival_distribution)
                    pickup_events = self.pickup_passengers(time_of_arrival, bus_id, current_stop_id, state)
                    new_events.extend(pickup_events)

                # No next stop but maybe has next trips? (will check in idle_update)
                if current_stop_number == last_stop_number:
                    state.buses[bus_id].current_stop_number = 0
                    state.buses[bus_id].status = BusStatus.IDLE
                    state.buses[bus_id].t_state_change = new_time

                    if len(state.buses[bus_id].bus_block_trips) > 0:
                        # time_of_activation = new_time
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
                            log(self.logger, state.time, f"Bus {bus_id} moves {distance:.2f} deadkms.", LogType.DEBUG)
                        else:
                            state.buses[bus_id].distance_to_next_stop = distance

                        time_to_state_change = time_of_activation + dt.timedelta(seconds=travel_time)
                        # Buses should start either at the scheduled time, or if they are late, should start as soon as possible.
                        time_to_state_change = max(time_to_state_change, scheduled_arrival_time)
                        state.buses[bus_id].t_state_change = time_to_state_change
                        state.buses[bus_id].time_at_last_stop = time_of_activation

                        arrival_event = Event(event_type=EventType.VEHICLE_ARRIVE_AT_STOP,
                                              time=time_to_state_change,
                                              type_specific_information={'bus_id': bus_id,
                                                                         'current_block_trip': current_block_trip,
                                                                         'stop': state.buses[bus_id].current_stop_number})
                        new_events.append(arrival_event)

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
                    elif scheduled_arrival_time >= time_to_state_change:
                        dwell_time = scheduled_arrival_time - time_to_state_change
                        state.buses[bus_id].dwell_time += dwell_time.total_seconds()
                        time_to_state_change = time_to_state_change + dwell_time

                    # HACK: This shouldn't happen (where a new event is earlier than the current time)
                    time_to_state_change = max(time_to_state_change, new_time)

                    state.buses[bus_id].t_state_change = time_to_state_change

                    # For distance
                    state.buses[bus_id].total_servicekms_moved += state.buses[bus_id].distance_to_next_stop
                    state.buses[bus_id].distance_to_next_stop = distance

                    arrival_event = Event(event_type=EventType.VEHICLE_ARRIVE_AT_STOP,
                                          time=time_to_state_change,
                                          type_specific_information={'bus_id': bus_id,
                                                                     'current_block_trip': current_block_trip,
                                                                     'stop': state.buses[bus_id].current_stop_number})
                    new_events.append(arrival_event)

            elif BusStatus.ALLOCATION == bus_state:
                pass

            elif BusStatus.BROKEN == bus_state:
                pass

        # Loop through all OVERLOAD buses
        for bus_id, bus_obj in state.buses.items():
            if (BusType.OVERLOAD == bus_obj.type) and (BusStatus.ALLOCATION == bus_obj.status):
                time_at_last_stop = bus_obj.time_at_last_stop
                t_state_change = bus_obj.t_state_change

                if new_time >= t_state_change:
                    bus_obj.status = BusStatus.IDLE
                    # Missing the last fraction of a distance
                    # bus_obj.total_deadkms_moved += (bus_obj.distance_to_next_stop - distance_fraction)
                    bus_obj.percent_to_next_stop = 0.0
                    bus_obj.distance_to_next_stop = 0
                    bus_obj.time_at_last_stop = new_time
                    bus_obj.partial_deadkms_moved = 0.0
                elif new_time < t_state_change:
                    if bus_obj.time_at_last_stop:
                        journey_fraction = (new_time - time_at_last_stop) / (t_state_change - time_at_last_stop)
                    else:
                        journey_fraction = 0.0
                    bus_obj.percent_to_next_stop = journey_fraction
                    distance_fraction = (journey_fraction * bus_obj.distance_to_next_stop)
                    bus_obj.total_deadkms_moved += (distance_fraction - bus_obj.partial_deadkms_moved)
                    bus_obj.partial_deadkms_moved = distance_fraction
                    log(self.logger, state.time,
                        f"Bus {bus_id} current total deadkms: {bus_obj.total_deadkms_moved:.2f}", LogType.DEBUG)

                    log(self.logger, new_time,
                        f"Bus {bus_id}: {journey_fraction * 100:.2f}% to {distance_fraction:.2f}/{bus_obj.distance_to_next_stop:.2f} kms to {bus_obj.current_stop_number}")

        state.time = new_time
        return new_events

    # TODO: Bug when overwriting trips with the same route_id_name
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
            passenger_waiting = {}
        if route_id_dir not in passenger_waiting:
            passenger_waiting[route_id_dir] = {}
        if scheduled_arrival_time not in passenger_waiting[route_id_dir]:
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

        if not passenger_waiting:
            return True

        picked_up_list = []
        for_deletion = []
        got_on_bus = 0
        ons = 0
        offs = 0
        remaining = 0

        new_events = []
        # TODO: Next time i should just use remaining -> ons
        if route_id_dir in passenger_waiting:
            for passenger_arrival_time, sampled_data in passenger_waiting[route_id_dir].items():
                # For some reason, some buses arrive earlier?
                if passenger_arrival_time > bus_arrival_time:
                    continue

                if bus_arrival_time - passenger_arrival_time <= dt.timedelta(minutes=PASSENGER_TIME_TO_LEAVE):
                    remaining = sampled_data['remaining']
                    sampled_ons = sampled_data['ons']
                    sampled_offs = sampled_data['offs']
                    got_on_bus = sampled_data.get("got_on_bus", 0)

                    # QUESTION: I think this needs to be += and not just = in case ons is non-zero.
                    if remaining > 0:
                        # sampled_ons += remaining
                        sampled_ons = remaining

                    if (got_on_bus == sampled_ons) and (sampled_ons > 0):
                        sampled_ons = 0

                    ons = sampled_ons
                    offs = sampled_offs

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

                    passenger_waiting[route_id_dir] = {
                        passenger_arrival_time: {'got_on_bus': got_on_bus, 'remaining': remaining,
                                                 'block_trip': current_block_trip,
                                                 'ons': ons, 'offs': offs}}
                    if remaining > 0:
                        log(self.logger,
                            _new_time, f"Bus {bus_id} left {remaining} people at stop {stop_id}",
                            LogType.ERROR)

                        # HACK full_state.time
                        # If someone is left behind, immediately flag a decision event
                        if full_state.buses[bus_id].type == BusType.REGULAR:
                            # _time = max(full_state.time, bus_arrival_time)
                            full_state.buses[bus_id].t_state_change = bus_arrival_time
                            event = Event(event_type=EventType.PASSENGER_LEFT_BEHIND,
                                          time=bus_arrival_time + dt.timedelta(seconds=1),
                                          type_specific_information={'bus_id': bus_id})
                            new_events.append(event)

                    stop_object.passenger_waiting[route_id_dir] = passenger_waiting[route_id_dir]
                    stop_object.total_passenger_ons += got_on_bus
                    stop_object.total_passenger_offs += offs

                    bus_object.current_load = bus_object.current_load + got_on_bus - offs
                    bus_object.total_passengers_served += got_on_bus
                    bus_object.total_stops += 1

                # Substitute for the leaving events
                elif passenger_arrival_time < (bus_arrival_time - dt.timedelta(minutes=PASSENGER_TIME_TO_LEAVE)):
                    sampled_ons = 0
                    sampled_offs = 0
                    walk_aways = 0
                    remaining = sampled_data.get("remaining", 0)
                    ons = sampled_data.get("ons", 0)

                    got_on_bus = sampled_data.get("got_on_bus", 0)
                    if remaining > 0:
                        walk_aways = remaining
                    stop_object.total_passenger_walk_away += walk_aways
                    for_deletion.append((route_id_dir, passenger_arrival_time))
                    ons = 0
                    offs = sampled_offs
                    got_on_bus = 0
                    if remaining > 0:
                        log(self.logger,
                            _new_time, f"{remaining} people left stop {stop_id}", LogType.ERROR)
                    remaining = 0
                    walk_aways = 0
                    passenger_waiting[route_id_dir] = {
                        passenger_arrival_time: {'got_on_bus': got_on_bus, 'remaining': remaining,
                                                 'block_trip': current_block_trip,
                                                 'ons': ons, 'offs': offs}}
                    if remaining > 0:
                        log(self.logger,
                            _new_time, f"Bus {bus_id} left {remaining} people at stop {stop_id}",
                            LogType.ERROR)

        log_str = f"""Bus {bus_id} on trip: {current_block_trip[1]} scheduled for {scheduled_arrival_time} \
arrives at @ {stop_id}: got_on:{got_on_bus:.0f}, on:{ons:.0f}, offs:{offs:.0f}, \
remain:{remaining:.0f}, bus_load:{bus_object.current_load:.0f}"""
        log(self.logger, _new_time, log_str, LogType.INFO)

        # for deletion in for_deletion:
        #     route_id_dir = deletion[0]
        #     time_key = deletion[1]
        #     if route_id_dir in stop_object.passenger_waiting:
        #         if time_key in stop_object.passenger_waiting[route_id_dir]:
        #             del stop_object.passenger_waiting[route_id_dir][time_key]

        return new_events

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
            current_block_trip = action_info[4]

            ofb_obj = state.buses[ofb_id]
            ofb_obj.bus_block_trips = [current_block_trip]
            stop_no = self.travel_model.get_stop_number_at_id(current_block_trip, stop_id)
            ofb_obj.current_stop_number = stop_no
            ofb_obj.status = BusStatus.IN_TRANSIT

            ofb_obj.current_block_trip = ofb_obj.bus_block_trips.pop(0)

            scheduled_arrival_time = self.travel_model.get_scheduled_arrival_time(current_block_trip,
                                                                                  stop_no)

            travel_time, distance = self.travel_model.get_traveltime_distance_from_depot(current_block_trip,
                                                                                         ofb_obj.current_stop,
                                                                                         stop_no)
            ofb_obj.total_deadkms_moved += distance
            log(self.logger, state.time, f"Bus {ofb_id} moves {distance:.2f} deadkms.", LogType.DEBUG)

            time_of_activation = state.time
            time_to_state_change = time_of_activation + dt.timedelta(seconds=travel_time)
            # Buses should start either at the scheduled time, or if they are late, should start as soon as possible.
            time_to_state_change = max(time_to_state_change, scheduled_arrival_time)
            ofb_obj.t_state_change = time_to_state_change
            ofb_obj.time_at_last_stop = time_of_activation

            event = Event(event_type=EventType.VEHICLE_ARRIVE_AT_STOP,
                          time=time_to_state_change,
                          type_specific_information={'bus_id': ofb_id,
                                                     'current_block_trip': current_block_trip,
                                                     'stop': state.buses[ofb_id].current_stop_number})

            new_events.append(event)

            log(self.logger, state.time,
                f"Dispatching overflow bus {ofb_id} from {ofb_obj.current_stop} @ stop {stop_id}",
                LogType.ERROR)

        # Take over broken bus
        elif ActionType.OVERLOAD_TO_BROKEN == action_type:
            ofb_obj = state.buses[ofb_id]
            action_info = action["info"]
            broken_bus_id = action_info
            broken_bus_obj = state.buses[broken_bus_id]

            current_block_trip = broken_bus_obj.current_block_trip
            stop_no = broken_bus_obj.current_stop_number

            ofb_obj.bus_block_trips = copy.copy([broken_bus_obj.current_block_trip] + broken_bus_obj.bus_block_trips)
            ofb_obj.bus_block_trips = [x for x in ofb_obj.bus_block_trips if x is not None]

            ofb_obj.current_block_trip = None
            # In case bus has not yet started trip.
            if stop_no == 0:
                ofb_obj.current_stop_number = 0
            # Because at this point we already set the state to the next stop.
            else:
                ofb_obj.current_stop_number = stop_no - 1

            ofb_obj.status = BusStatus.IN_TRANSIT

            # Switch passengers
            if copy.copy(broken_bus_obj.current_load) >= ofb_obj.capacity:
                ofb_obj.current_load = copy.copy(broken_bus_obj.current_load) - ofb_obj.capacity
            else:
                ofb_obj.current_load = copy.copy(broken_bus_obj.current_load)
            ofb_obj.total_passengers_served += ofb_obj.current_load

            # Deactivate broken_bus_obj
            broken_bus_obj.current_load = 0
            broken_bus_obj.current_block_trip = None
            broken_bus_obj.bus_block_trips = []
            broken_bus_obj.total_passengers_served -= ofb_obj.current_load

            # Prevent a late dispatch of bus to serve a very stale trip
            # trip_start_time = dt.datetime.combine(state.time.date(), dt.time(0, 0, 0))
            # moves = 0
            # while trip_start_time < state.time:
            #     if len(ofb_obj.bus_block_trips) > 0:
            #         current_block_trip = ofb_obj.bus_block_trips.pop(0)
            #         ofb_obj.current_block_trip = current_block_trip
            #         trip_start_time = self.travel_model.get_scheduled_arrival_time(current_block_trip, 0)
            #     else:
            #         return new_events, state.time
            #     moves += 1
            # if moves > 0:
            #     ofb_obj.current_stop_number = 0

            ofb_obj.current_block_trip = ofb_obj.bus_block_trips.pop(0)
            scheduled_arrival_time = self.travel_model.get_scheduled_arrival_time(ofb_obj.current_block_trip,
                                                                                  ofb_obj.current_stop_number)

            travel_time, distance = self.travel_model.get_traveltime_distance_from_depot(ofb_obj.current_block_trip,
                                                                                         ofb_obj.current_stop,
                                                                                         ofb_obj.current_stop_number)
            ofb_obj.total_deadkms_moved += distance
            log(self.logger, state.time, f"Bus {ofb_id} moves {distance:.2f} deadkms.", LogType.DEBUG)

            time_of_activation = state.time
            time_to_state_change = time_of_activation + dt.timedelta(seconds=travel_time)
            # HACK kind of.
            time_to_state_change = max(time_to_state_change, scheduled_arrival_time)
            ofb_obj.t_state_change = time_to_state_change

            event = Event(event_type=EventType.VEHICLE_ARRIVE_AT_STOP,
                          time=time_to_state_change,
                          type_specific_information={'bus_id': ofb_id,
                                                     'action': ActionType.OVERLOAD_TO_BROKEN})

            new_events.append(event)

            log(self.logger, state.time,
                f"Sending takeover overflow bus {ofb_id} from {ofb_obj.current_stop} @ stop {broken_bus_obj.current_stop}",
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
            ofb_obj.distance_to_next_stop = distance_to_next_stop
            ofb_obj.time_at_last_stop = state.time
            ofb_obj.status = BusStatus.ALLOCATION

        elif ActionType.NO_ACTION == action_type:
            # Do nothing
            pass

        return new_events, state.time
