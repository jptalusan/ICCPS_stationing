from Environment.enums import BusStatus, BusType

class SendNearestDispatchPolicy:
    
    def __init__(self, travel_model) -> None:
        self.travel_model = travel_model
        pass
    
    def get_overflow_bus_to_overflow_stop(self, state):
        for bus_id, bus_obj in state.buses.items():
            if bus_obj.type == BusType.OVERLOAD:
                if bus_obj.status == BusStatus.IDLE:
                    return bus_id
        return None