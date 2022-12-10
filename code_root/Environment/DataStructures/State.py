class State:
    
    def __init__(self,
                 stops,
                 buses,
                 time,
                 bus_events=[]):
        
        self.stops = stops
        self.buses = buses
        self.bus_events = bus_events
        self.time = time
        self.trips_with_px_left = {}
        self.served_trips = []