class State:
    
    def __init__(self,
                 stops,
                 buses,
                 time,
                 events=[]):
        
        self.stops = stops
        self.buses = buses
        self.events = events
        self.time = time