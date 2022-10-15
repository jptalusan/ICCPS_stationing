from src.utils import *
import json
import warnings
import pandas as pd
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


# For now should contain all travel related stuff (ons, loads, travel times, distances)
class EmpiricalTravelModelLookup:
    def __init__(self, logger):
        config_path = 'scenarios/baseline/data/config.json'
        with open(config_path) as f:
            config = dotdict(json.load(f))

        config_path = f'scenarios/baseline/data/{config.trip_plan}'
        with open(config_path) as f:
            self.trip_plan = dotdict(json.load(f))
            
        travel_time_path = 'scenarios/baseline/data/sampled_travel_times.pkl'
        self.sampled_travel_time = pd.read_pickle(travel_time_path)
        
        distance_path = 'scenarios/baseline/data/gtfs_distance_pairs_km.pkl'
        self.sampled_distance = pd.read_pickle(distance_path)
        
        disruption_path = 'scenarios/baseline/data/disruption_probabilities.pkl'
        self.sampled_disruption = pd.read_pickle(disruption_path)
        
        self.logger = logger

        lookup_path = 'scenarios/baseline/data/pair_tt_dd_stops.pkl'
        self.lookup_tt_dd = pd.read_pickle(lookup_path)
    
    # pandas dataframe: route_id_direction, block_abbr, stop_id_original, time, IsWeekend, sample_time_to_next_stop
    def get_travel_time(self, current_block_trip, current_stop_number, _datetime):
        current_trip       = current_block_trip[1]
        IsWeekend          = 0 if _datetime.weekday() < 5 else 1
        # time               = _datetime.time().strftime('%H:%M:%S')
        block_abbr         = int(current_block_trip[0])
        
        trip_data          = self.trip_plan[current_trip]
        route_id           = trip_data['route_id']
        route_direction    = trip_data['route_direction']
        route_id_dir       = str(route_id) + "_" + route_direction
        stop_id_original   = trip_data['stop_id_original']
        stop_id_original   = stop_id_original[current_stop_number]
        
        scheduled_time     = self.trip_plan[current_trip]['scheduled_time'][current_stop_number].split(" ")[1]
        
        # print(scheduled_time, route_id_dir, block_abbr, current_stop_number + 1, stop_id_original, IsWeekend)
        tdf = self.sampled_travel_time[(self.sampled_travel_time['route_id_direction'] == route_id_dir) & \
                                       (self.sampled_travel_time['block_abbr'] == block_abbr) & \
                                       (self.sampled_travel_time['stop_sequence'] == current_stop_number + 1) & \
                                       (self.sampled_travel_time['stop_id_original'] == stop_id_original) & \
                                       (self.sampled_travel_time['IsWeekend'] == IsWeekend)]
        tdf['time'] = pd.to_datetime(tdf['time'], format='%H:%M:%S')
        
        tdf = tdf[tdf['time'] == f'1900-01-01 {scheduled_time}']
        if not tdf.empty:
            return tdf.iloc[0]['sampled_travel_time']
        
        # TODO: Handle when not available!
        log(self.logger, _datetime, f'Failed to get travel time for: {scheduled_time},{route_id_dir},{block_abbr},{current_stop_number+1},{stop_id_original},{IsWeekend}', LogType.ERROR)
        return self.get_travel_time_from_depot(current_block_trip, stop_id_original, current_stop_number, _datetime)
    
    def get_travel_time_from_depot(self, current_block_trip, current_stop, current_stop_number, _datetime):
        current_trip = current_block_trip[1]
        trip_data = self.trip_plan[current_trip]
        next_stop = trip_data['stop_id_original'][current_stop_number]

        if (current_stop is not None) and (current_stop == next_stop):
            return 0
        
        tt = self.lookup_tt_dd.query("current_stop == @current_stop and next_stop == @next_stop")
        if not tt.empty:
            return tt.iloc[0]['travel_time_s']
        else:
            print(f"Travel time cannot be computed for {current_stop} and {next_stop}")
            raise "Error getting Travel time"
    
    # tt is in seconds
    def get_travel_time_from_stop_to_stop(self, current_stop, next_stop, _datetime):
        if (current_stop is not None) and (current_stop == next_stop):
            return 0
        
        tt = self.lookup_tt_dd.query("current_stop == @current_stop and next_stop == @next_stop")
        if not tt.empty:
            return tt.iloc[0]['travel_time_s']
        else:
            print(f"Travel time cannot be computed for {current_stop} and {next_stop}")
            raise "Error getting Travel time"
    
    # dd is in meters
    def get_distance_from_stop_to_stop(self, current_stop, next_stop, _datetime):
        if current_stop and next_stop:
            if (current_stop == next_stop):
                return 0
            
            dd = self.lookup_tt_dd.query("current_stop == @current_stop and next_stop == @next_stop")
            if not dd.empty:
                return dd.iloc[0]['distance_m'] / 1000
            else:
                dd = self.sampled_distance.query("stop_id == @current_stop and next_stop_id == @next_stop")
                if not dd.empty:
                    return dd.iloc[0]['shape_dist_traveled_km']
                else:
                    print(f"Distance cannot be computed for {current_stop} and {next_stop}")
                    raise "Error getting distance"
        else:
            return -1
            
    # pandas dataframe: stop_id, next_stop_id, shape_dist_traveled_km
    def get_distance(self, current_stop, next_stop, _datetime):
        if current_stop and next_stop:
            if (current_stop == next_stop):
                return 0
            
            dd = self.lookup_tt_dd.query("current_stop == @current_stop and next_stop == @next_stop")
            if not dd.empty:
                return dd.iloc[0]['distance_m'] / 1000
            else:
                dd = self.sampled_distance.query("stop_id == @current_stop and next_stop_id == @next_stop")
                if not dd.empty:
                    return dd.iloc[0]['shape_dist_traveled_km']
                else:
                    print(f"Distance cannot be computed for {current_stop} and {next_stop}")
                    raise "Error getting distance"
        else:
            print(f"Distance cannot be computed for {current_stop} and {next_stop}")
            raise "Error getting distance"
            return -1
    
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
        route_id           = trip_data['route_id']
        route_direction    = trip_data['route_direction']
        return str(route_id) + "_" + route_direction
        
    def get_last_stop_number_on_trip(self, current_block_trip):
        current_trip = current_block_trip[1]
        trip_data = self.trip_plan[current_trip]
        return trip_data['last_stop_sequence']
        