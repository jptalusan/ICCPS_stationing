from Environment.enums import BusStatus, BusType
import random

class RandomDispatch:
    
    def __init__(self, travel_model) -> None:
        self.travel_model = travel_model
        pass
    
    def get_overflow_bus_to_overflow_stop(self, state):
        buses = []
        for bus_id, bus_obj in state.buses.items():
            if bus_obj.type == BusType.OVERLOAD:
                if bus_obj.status == BusStatus.IDLE:
                    buses.append(bus_id)
        if len(buses) > 0:
            return random.choice(buses)
        return None