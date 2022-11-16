from src.utils import *
import json
import warnings
import pickle
import pandas as pd
import datetime as dt
from pandas.core.common import SettingWithCopyWarning
from Environment.enums import EventType, ActionType, BusType

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


# For now should contain all travel related stuff (ons, loads, travel times, distances)
class EmpiricalTravelModelLookup:
    def __init__(self, date_str, logger):
        config_path = f'scenarios/baseline/data/trip_plan_{date_str}.json'
        with open(config_path) as f:
            self.trip_plan = json.load(f)

        disruption_path = 'scenarios/baseline/data/disruption_probabilities.pkl'
        self.sampled_disruption = pd.read_pickle(disruption_path)

        self.logger = logger

        with open('scenarios/baseline/data/sampled_travel_times_dict.pkl', 'rb') as handle:
            self.sampled_travel_time = pickle.load(handle)

        with open('scenarios/baseline/data/stops_tt_dd_node_dict.pkl', 'rb') as handle:
            self.stops_tt_dd_dict = pickle.load(handle)

        with open('scenarios/baseline/data/stops_node_matching_dict.pkl', 'rb') as handle:
            self.stop_nodes_dict = pickle.load(handle)

        with open(f'scenarios/baseline/data/time_point_dict_{date_str}.pkl', 'rb') as handle:
            self.time_point_dict = pickle.load(handle)

    # pandas dataframe: route_id_direction, block_abbr, stop_id_original, time, IsWeekend, sample_time_to_next_stop
    def get_travel_time(self, current_block_trip, current_stop_number, _datetime):
        current_trip = current_block_trip[1]
        IsWeekend = 0 if _datetime.weekday() < 5 else 1
        block_abbr = int(current_block_trip[0])

        trip_data = self.trip_plan[current_trip]
        route_id = trip_data['route_id']
        route_direction = trip_data['route_direction']
        route_id_dir = str(route_id) + "_" + route_direction
        stop_id_original = trip_data['stop_id_original']
        stop_id_original = stop_id_original[current_stop_number]

        scheduled_time = self.trip_plan[current_trip]['scheduled_time'][current_stop_number].split(" ")[1]
        scheduled_time = dt.datetime.strptime(scheduled_time, "%H:%M:%S").time()
        # rid, bid, ssq, sid, tme, wkd
        key = (route_id_dir, block_abbr, current_stop_number + 1, stop_id_original, scheduled_time, IsWeekend)

        if key in self.sampled_travel_time:
            tt = self.sampled_travel_time[key]['sampled_travel_time']
            return tt

        # log(self.logger, _datetime, f'Failed to get travel time for: {key}', LogType.ERROR)
        return self.get_travel_time_from_depot(current_block_trip, stop_id_original, current_stop_number, _datetime)

    def get_travel_time_from_depot(self, current_block_trip, current_stop, current_stop_number, _datetime):
        current_trip = current_block_trip[1]
        trip_data = self.trip_plan[current_trip]
        next_stop = trip_data['stop_id_original'][current_stop_number]

        tt, _ = self.get_traveltime_distance_from_stops(current_stop, next_stop)
        return tt

    # tt is in seconds
    def get_travel_time_from_stop_to_stop(self, current_stop, next_stop, _datetime):
        tt, _ = self.get_traveltime_distance_from_stops(current_stop, next_stop)
        return tt

    # dd is in meters
    def get_distance_from_stop_to_stop(self, current_stop, next_stop, _datetime):
        _, dd = self.get_traveltime_distance_from_stops(current_stop, next_stop)
        return dd

    # Can probably move to a different file next time
    def get_next_stop(self, current_block_trip, current_stop_sequence):
        '''
        trip_plan[trip_id][stop_id_original]
        
        '''
        current_trip = current_block_trip[1]
        trip_data = self.trip_plan[current_trip]
        last_stop_id = trip_data['last_stop_id']
        last_stop_sequence = trip_data['last_stop_sequence']

        if current_stop_sequence == None:
            return None

        if current_stop_sequence == last_stop_sequence:
            return None

        else:
            return trip_data['stop_sequence'][current_stop_sequence + 1]

    def get_stop_id_at_number(self, current_block_trip, current_stop_sequence):
        if current_stop_sequence == -1:
            return None
        current_trip = current_block_trip[1]
        trip_data = self.trip_plan[current_trip]
        return trip_data['stop_id_original'][current_stop_sequence]

    def get_stop_number_at_id(self, current_block_trip, current_stop_id):
        current_trip = current_block_trip[1]
        trip_data = self.trip_plan[current_trip]
        stop_idx = trip_data['stop_id_original'].index(current_stop_id)
        return trip_data['stop_sequence'][stop_idx]

    def get_scheduled_arrival_time(self, current_block_trip, current_stop_sequence):
        current_trip = current_block_trip[1]
        trip_data = self.trip_plan[current_trip]
        arrival_time_str = trip_data['scheduled_time'][current_stop_sequence]
        scheduled_arrival_time = str_timestamp_to_datetime(arrival_time_str)
        return scheduled_arrival_time

    def get_disruption_probability(self, stop_id):
        tdf = self.sampled_disruption.query("stop_id_original == @stop_id")
        if not tdf.empty:
            return tdf.iloc[0].probability
        return 0

    def get_route_id_dir_for_trip(self, current_block_trip):
        current_trip = current_block_trip[1]
        trip_data = self.trip_plan[current_trip]
        route_id = trip_data['route_id']
        route_direction = trip_data['route_direction']
        return str(route_id) + "_" + route_direction

    def get_last_stop_number_on_trip(self, current_block_trip):
        current_trip = current_block_trip[1]
        trip_data = self.trip_plan[current_trip]
        return trip_data['last_stop_sequence']

    def get_traveltime_distance_from_depot(self, current_block_trip, current_stop, current_stop_number):
        dd = -1
        tt = -1
        current_trip = current_block_trip[1]
        trip_data = self.trip_plan[current_trip]
        next_stop = trip_data['stop_id_original'][current_stop_number]

        tt, dd = self.get_traveltime_distance_from_stops(current_stop, next_stop)
        return tt, dd

    def get_traveltime_distance_from_stops(self, current_stop, next_stop):
        cn = self.stop_nodes_dict[current_stop]['nearest_node']
        nn = self.stop_nodes_dict[next_stop]['nearest_node']

        if (cn, nn) in self.stops_tt_dd_dict:
            tt = self.stops_tt_dd_dict[(cn, nn)]['travel_time_s']
            dd = self.stops_tt_dd_dict[(cn, nn)]['distance_m']
        elif (nn, cn) in self.stops_tt_dd_dict:
            tt = self.stops_tt_dd_dict[(nn, cn)]['travel_time_s']
            dd = self.stops_tt_dd_dict[(nn, cn)]['distance_m']
        else:
            # print(f"Could not find tt/dd for {cn}/{current_stop} to {nn}/{next_stop}.")
            return 60, 0.5
            # raise "Error"
        if tt < 0:
            # print(f"Could not find tt/dd for {cn}/{current_stop} to {nn}/{next_stop}.")
            return 60, 0.5
            # raise "Error"
        return tt, dd / 1000

    def get_last_arrival_event(self, state):
        trips = []
        for bus_id, bus_obj in state.buses.items():
            for block_trip in bus_obj.bus_block_trips:
                trips.append(block_trip[1])

        last_trip_arrival = dt.datetime(year=2000, month=1, day=1, hour=0, minute=0, second=0)
        for trip in trips:
            trip_data = self.trip_plan[trip]
            lta = trip_data['scheduled_time'][-1]
            lta = str_timestamp_to_datetime(lta)

            if lta >= last_trip_arrival:
                last_trip_arrival = lta

        return last_trip_arrival

    def get_list_of_stops_for_trip(self, trip, current_stop_number):
        trip_data = self.trip_plan[trip]
        stop_id_original = trip_data['stop_id_original']
        stop_id_original = stop_id_original[0:current_stop_number+1]
        return stop_id_original

    def is_event_a_timepoint(self, curr_event, state):
        info = curr_event.type_specific_information
        if curr_event.event_type == EventType.VEHICLE_BREAKDOWN:
            return True

        if curr_event.event_type == EventType.VEHICLE_ARRIVE_AT_STOP:
            # bus_id = info['bus_id']
            current_block_trip = info['current_block_trip']
            current_trip = current_block_trip[1]
            current_stop_number = info['stop']
            # bus_object = state.buses[bus_id]
            current_stop_id = self.get_stop_id_at_number(current_block_trip, current_stop_number)
            scheduled_time = str_timestamp_to_datetime(self.trip_plan[current_trip]['scheduled_time'][current_stop_number])

            key = (int(current_trip), current_stop_id, scheduled_time)
            timepoint = self.time_point_dict[key]['timepoint']
            return timepoint == 1

        return False