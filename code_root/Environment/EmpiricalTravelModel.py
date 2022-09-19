from src.utils import *
import json
import pandas as pd

# For now should contain all travel related stuff (ons, loads, travel times, distances)
class EmpiricalTravelModel:
    def __init__(self, logger):
        config_path = 'scenarios/baseline/data/config.json'
        with open(config_path) as f:
            config = dotdict(json.load(f))

        config_path = f'scenarios/baseline/data/{config.trip_plan}'
        with open(config_path) as f:
            self.trip_plan = dotdict(json.load(f))
            
        travel_time_path = '/home/jptalusan/gits/mta_simulator_redo/code_root/scenarios/baseline/data/sampled_travel_times.pkl'
        self.sampled_travel_time = pd.read_pickle(travel_time_path)
        
        distance_path = '/home/jptalusan/gits/mta_simulator_redo/code_root/scenarios/baseline/data/gtfs_distance_pairs_km.pkl'
        self.sampled_distance = pd.read_pickle(distance_path)
        
        self.logger = logger
        pass
    
    # pandas dataframe: route_id_direction, block_abbr, stop_id_original, time, IsWeekend, sample_time_to_next_stop
    def get_travel_time(self, current_block_trip, current_stop, _datetime):
        IsWeekend = 0 if _datetime.weekday() < 5 else 1
        time = _datetime.time().strftime('%H:%M:%S')
        block_abbr = current_block_trip[0]
        trip = current_block_trip[1]
        stop_id_original = current_stop
        
        pass
    
    #TODO: Add OSM computation
    def get_travel_time_from_depot(self):
        return 100
    
    # pandas dataframe: stop_id, next_stop_id, shape_dist_traveled_km
    def get_distance(self):
        pass
    
    # Can probably move to a different file next time
    def get_next_stop(self, current_block_trip, current_stop_sequence):
        '''
        trip_plan[trip_id][stop_id_original]
        
        '''
        current_trip = current_block_trip[1]
        trip_data = self.trip_plan[current_trip]
        last_stop_id = trip_data['last_stop_id']
        last_stop_sequence = trip_data['last_stop_sequence']
        
        if current_stop_sequence == last_stop_sequence:
            return None
        
        else:
            return trip_data['stop_sequence'][current_stop_sequence + 1]