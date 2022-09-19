import copy
import datetime as dt
from Environment.enums import BusStatus
from src.utils import *

class BusDynamics:
    
    def __init__(self, travel_model, logger) -> None:
        self.travel_model = travel_model
        self.logger = logger
    
    def update_bus(self, bus_id, _new_time, full_state):
        '''
        In this function, a responder's state will be updated to the given time. The responder can pass through
        multiple status changes during this update
        :param resp_obj:
        :param resp_id:
        :param _new_time:
        :param full_state:
        :return:
        '''

        curr_bus_time = copy.copy(full_state.time)

        # update the responder until it catches up with new time
        while curr_bus_time < _new_time:
            resp_status = full_state.buses[bus_id].status

            new_resp_time = None
            
            if resp_status == BusStatus.BROKEN:
                new_resp_time, update_status = self.broken_update(bus_id, curr_bus_time, _new_time, full_state)
            elif resp_status == BusStatus.IDLE:
                new_resp_time, update_status = self.idle_update(bus_id, curr_bus_time, _new_time, full_state)
            elif resp_status == BusStatus.IN_TRANSIT:
                new_resp_time, update_status = self.transit_update(bus_id, curr_bus_time, _new_time, full_state)
                
            curr_bus_time = new_resp_time
        
        log(self.logger, curr_bus_time, _new_time)
        if curr_bus_time == _new_time:
            resp_status = full_state.buses[bus_id].status

            new_resp_time = None
            
            if resp_status == BusStatus.BROKEN:
                new_resp_time, update_status = self.broken_update(bus_id, curr_bus_time, _new_time, full_state)
            elif resp_status == BusStatus.IDLE:
                new_resp_time, update_status = self.idle_update(bus_id, curr_bus_time, _new_time, full_state)
            elif resp_status == BusStatus.IN_TRANSIT:
                new_resp_time, update_status = self.transit_update(bus_id, curr_bus_time, _new_time, full_state)
                
            curr_bus_time = new_resp_time
            
    def broken_update(self, bus_id, curr_bus_time, _new_time, full_state):
        return _new_time, False
            
    def idle_update(self, bus_id, curr_bus_time, _new_time, full_state):
        self.logger.debug("Enter idle_update()")
        '''
        In this case, there is nothing to update for the bus' state, since it is waiting (between trips, before trips).
        '''
        if _new_time >= full_state.buses[bus_id].t_state_change:
            time_of_activation = full_state.buses[bus_id].t_state_change
            full_state.buses[bus_id].status = BusStatus.IN_TRANSIT
            # Move trips
            if full_state.buses[bus_id].next_block_trip:
                print("HERE")
                current_block_trip = full_state.buses[bus_id].next_block_trip
                bus_block_trips = full_state.buses[bus_id].bus_block_trips
                for i, block_trips in enumerate(bus_block_trips):
                    if (current_block_trip == block_trips) and (i == len(bus_block_trips) - 1):
                        full_state.buses[bus_id].next_block_trip = None
                    else:
                        full_state.buses[bus_id].next_block_trip = bus_block_trips[i + 1]
                    
                current_stop_number = full_state.buses[bus_id].current_stop_number
                full_state.buses[bus_id].next_stop_number = self.travel_model.get_next_stop(current_block_trip, current_stop_number)
                
                travel_time = self.travel_model.get_travel_time_from_depot()
                full_state.buses[bus_id].t_state_change = time_of_activation + dt.timedelta(seconds=travel_time)
                log(self.logger, curr_bus_time, full_state.buses[bus_id].t_state_change)
                print("EHER", travel_time, time_of_activation, type(time_of_activation), time_of_activation + dt.timedelta(seconds=travel_time))
                
                return time_of_activation, True
            # no more trips left
            else:
                pass
        
        return _new_time, False
            
    def transit_update(self, bus_id, curr_bus_time, _new_time, full_state):
        log(self.logger, curr_bus_time, "enter transit_update")
        '''
        In this case, we update the position of the bus
        update current stop number <- next_stop_number
        update next_stop_number <- new next stop
        '''
        if _new_time >= full_state.buses[bus_id].t_state_change:
            pass
        
        # Interpolate
        else:
            journey_fraction = (_new_time - curr_bus_time) / (full_state.buses[bus_id].t_state_change - curr_bus_time)
            log(self.logger, curr_bus_time, journey_fraction)
        return _new_time, False
            
    
    