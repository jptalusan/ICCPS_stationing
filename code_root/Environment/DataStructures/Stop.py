class Stop:
    
    def __init__(self,
                 stop_id,
                 total_passenger_ons,
                 total_passenger_offs,
                 total_passenger_walk_away,
                 passenger_waiting):
        
        self.stop_id = stop_id
        self.total_passenger_ons = total_passenger_ons
        self.total_passenger_offs = total_passenger_offs
        self.total_passenger_walk_away = total_passenger_walk_away
        self.passenger_waiting = passenger_waitings