# Current_stop is the starting depot if the current_stop_number is 0
class Bus:
    def __init__(self,
                 my_id,
                 type,
                 status,
                 capacity,
                 bus_block_trips=[],
                 current_stop="",
                 current_block_trip=None,
                 next_block_trip=None,
                 current_stop_number=-1,
                 next_stop_number=0,
                 t_state_change=0.0,
                 percent_to_next_stop=0.0,
                 total_distance_moved=0,
                 total_passengers_served=0,
                 available_time=0.0,
                 dwell_time=0.0,
                 delay_time=0.0):
        self.my_id = my_id
        self.status = status
        self.type = type
        self.bus_block_trips = bus_block_trips
        self.current_block_trip = current_block_trip
        self.next_block_trip = next_block_trip
        self.current_stop = current_stop
        self.current_stop_number = current_stop_number
        self.next_stop_number = next_stop_number
        self.t_state_change = t_state_change
        self.percent_to_next_stop = percent_to_next_stop
        self.total_distance_moved = total_distance_moved
        self.total_passengers_served = total_passengers_served
        self.available_time = available_time
        self.dwell_time = dwell_time
        self.delay_time = delay_time
        self.capacity = capacity

    def __str__(self):
        return f"{self.my_id},{self.status},{self.current_block_trip}"
            
    def __repr__(self):
        return f"{self.my_id},{self.status},{self.current_block_trip}"