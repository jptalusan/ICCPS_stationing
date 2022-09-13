from Environment import enums
from Environment.BusDynamics import BusDynamics
from Environment.StopDynamics import StopDynamics

class EnvironmentModel:
    
    def __init__(self, travel_model) -> None:
        self.bus_dynamics = BusDynamics()
        self.stop_dynamics = StopDynamics()
        self.travel_model = travel_model
        
        