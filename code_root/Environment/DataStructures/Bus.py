class Bus:
    def __init__(self,
                 my_id,
                 type,
                 status,
                 trips,
                 current_trip,
                 next_trip,
                 current_stop_sequence,
                 next_stop_sequence,
                 percent_to_next_stop,
                 total_distance_moved,
                 total_passengers_served,
                 available_time,
                 dwell_time,
                 delay_time):
        self.my_id = my_id
        self.status = status
        self.type = type
        self.trips = trips
        self.current_trip = current_trip
        self.next_trip = next_trip
        self.current_stop_number = current_stop_sequence
        self.next_stop_number = next_stop_sequence
        self.percent_to_next_stop = percent_to_next_stop
        self.total_distance_moved = total_distance_moved
        self.total_passengers_served = total_passengers_served
        self.available_time = available_time
        self.dwell_time = dwell_time
        self.delay_time = delay_time