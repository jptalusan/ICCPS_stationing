from Environment import enums
from Environment.BusDynamics import BusDynamics
from Environment.StopDynamics import StopDynamics
from Environment.enums import BusStatus, EventType
from src.utils import *

class EnvironmentModel:
    
    def __init__(self, travel_model, logger) -> None:
        self.bus_dynamics = BusDynamics(travel_model, logger)
        self.stop_dynamics = StopDynamics(travel_model, logger)
        self.travel_model = travel_model
        self.logger = logger
        
    def update(self, state, curr_event):
        '''
        Updates the state to the given time. This is mostly updating the responders
        :param state:
        :param new_time:
        :return:
        '''
        new_events = []
        new_time = curr_event.time
        # print(new_time, state.time)
        assert new_time >= state.time
        
        # if (curr_event.event_type == EventType.VEHICLE_START_TRIP) or \
        #    (curr_event.event_type == EventType.VEHICLE_ARRIVE_AT_STOP) or \
        #    (curr_event.event_type == EventType.VEHICLE_ACCIDENT) or \
        #    (curr_event.event_type == EventType.VEHICLE_BREAKDOWN):
        log(self.logger, new_time, curr_event, LogType.DEBUG)
        
        if (curr_event.event_type == EventType.PASSENGER_ARRIVE_STOP) or \
           (curr_event.event_type == EventType.PASSENGER_LEAVE_STOP):
            # update the state of A stop
            _new_events = self.stop_dynamics.update_stop(curr_event,
                                           _new_time=new_time,
                                           full_state=state)
            new_events.extend(_new_events)

            # update the state of EACH bus
        for bus_id, bus_obj in state.buses.items():
            _new_events = self.bus_dynamics.update_bus(bus_id=bus_id,
                                                        _new_time=new_time,
                                                        full_state=state)
            new_events.extend(_new_events)
            
        # for stop_id, stop_obj in state.stops.items():
        #     print(stop_obj)
        state.time = new_time
        return new_events