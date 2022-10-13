class State:
    
    def __init__(self,
                 stops,
                 buses,
                 time,
                 events=[],
                 active_incidents=[]):
        
        self.stops = stops
        self.buses = buses
        self.events = events
        self.time = time
        
        self.active_incidents = active_incidents