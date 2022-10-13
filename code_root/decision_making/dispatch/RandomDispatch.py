from Environment.enums import BusStatus, BusType
import random

class RandomDispatch:
    
    def __init__(self, travel_model) -> None:
        self.travel_model = travel_model
        pass
    
    def select_overload_to_dispatch(self, state, actions):
        return random.choice(actions)