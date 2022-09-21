import copy
import datetime as dt
from Environment.enums import BusStatus, EventType
from Environment.DataStructures.Event import Event
from src.utils import *

class BusDynamics:
    
    def __init__(self, travel_model, logger) -> None:
        self.travel_model = travel_model
        self.logger = logger
        self.new_events = []
    
    def update_bus(self, bus_id, _new_time, full_state):
        '''
        In this function, a responder's state will be updated to the given time. The responder can pass through
        multiple status changes during this update
        :param resp_obj:
        :param resp_id:
        :param _new_time: current event's time
        :param full_state:
        :return:
        '''
        self.new_events = []
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
        
        # log(self.logger, curr_bus_time, f"Before if: {curr_bus_time}, {_new_time}")
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
            
        return self.new_events
    
    def broken_update(self, bus_id, curr_bus_time, _new_time, full_state):
        full_state.buses[bus_id].status = BusStatus.BROKEN
        
        return _new_time, False
            
    def idle_update(self, bus_id, curr_bus_time, _new_time, full_state):
        self.logger.debug("Enter idle_update()")
        '''
        In this case, there is nothing to update for the bus' state, since it is waiting (between trips, before trips).
        '''
        # Activate bus if idle and has trips
        if _new_time >= full_state.buses[bus_id].t_state_change:
            time_of_activation = full_state.buses[bus_id].t_state_change
            # Move trips
            if full_state.buses[bus_id].next_block_trip:
                full_state.buses[bus_id].status = BusStatus.IN_TRANSIT
                current_block_trip = full_state.buses[bus_id].next_block_trip
                full_state.buses[bus_id].current_block_trip = current_block_trip
                bus_block_trips = full_state.buses[bus_id].bus_block_trips
                
                bbt_idx = bus_block_trips.index(current_block_trip)
                if bbt_idx == (len(bus_block_trips) - 1) or len(bus_block_trips) == 1:
                    full_state.buses[bus_id].next_block_trip = None
                else:
                    full_state.buses[bus_id].next_block_trip = bus_block_trips[bbt_idx + 1]
                    
                full_state.buses[bus_id].current_stop_number = 0
                full_state.buses[bus_id].next_stop_number = self.travel_model.get_next_stop(current_block_trip, 
                                                                                            full_state.buses[bus_id].current_stop_number)
                
                travel_time = self.travel_model.get_travel_time_from_depot()
                time_to_state_change = time_of_activation + dt.timedelta(seconds=travel_time)
                full_state.buses[bus_id].t_state_change = time_to_state_change
                
                event = Event(event_type=EventType.VEHICLE_ARRIVE_AT_STOP, 
                              time=time_to_state_change,
                              type_specific_information={'bus_id': bus_id, 
                                                         'stop':0})
                self.new_events.append(event)
                return time_of_activation, True
            # no more trips left
            else:
                pass
        # No trips in the future
        else:
            full_state.buses[bus_id].status = BusStatus.IDLE            
            pass
        
        return _new_time, False
            
    def transit_update(self, bus_id, curr_bus_time, _new_time, full_state):
        '''
        In this case, we update the position of the bus
        update current stop number <- next_stop_number
        update next_stop_number <- new next stop
        '''
        # log(self.logger, _new_time, f"bus time:{curr_bus_time}, bus state change:{full_state.buses[bus_id].t_state_change}")
        
        # Calculate travel time to next stop and update t_state_change
        if _new_time >= full_state.buses[bus_id].t_state_change:
            time_of_arrival        = full_state.buses[bus_id].t_state_change
            current_block_trip     = full_state.buses[bus_id].current_block_trip
            bus_block_trips     = full_state.buses[bus_id].bus_block_trips
            current_stop_number    = full_state.buses[bus_id].current_stop_number
            next_stop_number       = full_state.buses[bus_id].next_stop_number
            scheduled_arrival_time = self.travel_model.get_scheduled_arrival_time(current_block_trip, current_stop_number)
            current_stop_id        = self.travel_model.get_stop_id_at_number(current_block_trip, current_stop_number)
            
            log(self.logger, _new_time, f"Bus {bus_id} on trip: {current_block_trip[1]} scheduled for {scheduled_arrival_time} arrives at stop: {current_stop_id}", LogType.INFO)
            log(self.logger, _new_time, f"bus:{bus_block_trips},{next_stop_number}")
            # Bus running time
            if full_state.buses[bus_id].time_at_last_stop:
                full_state.buses[bus_id].total_service_time += (_new_time - full_state.buses[bus_id].time_at_last_stop).total_seconds()
                
            full_state.buses[bus_id].time_at_last_stop = time_of_arrival
            
            # If valid stop
            if full_state.buses[bus_id].current_stop_number >= 0:
                self.pickup_passengers(_new_time, bus_id, current_stop_id, full_state)
            
            # No next stop
            if next_stop_number == None:
                full_state.buses[bus_id].status = BusStatus.IDLE
            # Going to next stop
            else:
                full_state.buses[bus_id].current_stop_number = next_stop_number
                full_state.buses[bus_id].next_stop_number    = self.travel_model.get_next_stop(current_block_trip, next_stop_number)
                travel_time            = self.travel_model.get_travel_time(current_block_trip, current_stop_number, _new_time)
                scheduled_arrival_time = self.travel_model.get_scheduled_arrival_time(current_block_trip, next_stop_number)
                
                time_to_state_change = time_of_arrival + dt.timedelta(seconds=travel_time)
                
                # Taking into account delay time
                if scheduled_arrival_time < time_to_state_change:
                    delay_time = time_to_state_change - scheduled_arrival_time
                    log(self.logger, _new_time, f"delay: {delay_time.total_seconds()}")
                    full_state.buses[bus_id].delay_time += delay_time.total_seconds()
                    
                # TODO: Not the best place to put this, Dwell time
                elif scheduled_arrival_time > time_to_state_change:
                    dwell_time = scheduled_arrival_time - time_to_state_change
                    log(self.logger, _new_time, f"dwell: {dwell_time.total_seconds()}")
                    full_state.buses[bus_id].dwell_time += dwell_time.total_seconds()
                    time_to_state_change = time_to_state_change + dwell_time
                
                full_state.buses[bus_id].t_state_change = time_to_state_change
                
                event = Event(event_type=EventType.VEHICLE_ARRIVE_AT_STOP, 
                              time=time_to_state_change,
                              type_specific_information={'bus_id': bus_id, 
                                                         'stop':next_stop_number})
                self.new_events.append(event)
                # log(self.logger, _new_time, f"Creating new event: {event}")
            
            return time_of_arrival, False
        
        # Interpolate
        else:
            if full_state.buses[bus_id].time_at_last_stop:
                journey_fraction = (_new_time - full_state.buses[bus_id].time_at_last_stop) / (full_state.buses[bus_id].t_state_change - full_state.buses[bus_id].time_at_last_stop)
            else:
                journey_fraction = 0.0
            log(self.logger, _new_time, f"Fraction: {journey_fraction:.2f} to {full_state.buses[bus_id].current_stop_number}")
            full_state.buses[bus_id].percent_to_next_stop = journey_fraction
            return _new_time, False
            
    # TODO: Handle zero_load_at_trip_end
    def pickup_passengers(self, _new_time, bus_id, stop_id, full_state):
        bus_object = full_state.buses[bus_id]
        stop_object = full_state.stops[stop_id]
        vehicle_capacity = bus_object.capacity
        passenger_waiting = stop_object.passenger_waiting
        
        # Passenger load, board, alight computations
        current_load = bus_object.current_load
        
        sampled_load = 0
        sampled_ons = 0
        # TODO: How to handle passengers that have been waiting for a time, passengers in different route and direction?
        if passenger_waiting:
            key = None
            for passenger_arrival, sampled_data in passenger_waiting.items():
                if passenger_arrival <= _new_time:
                    key = passenger_arrival
                    sampled_load = sampled_data['load']
                    sampled_ons  = sampled_data['ons']
                    bLeft = sampled_data['bLeft']
                    break
                
            if current_load > sampled_load:
                offs = ((current_load - sampled_load) / current_load) * current_load
                ons  = sampled_ons
            elif current_load < sampled_load:
                ons  = sampled_load - current_load + sampled_ons
                offs = 0
            else:
                ons  = 0
                offs = 0
            left_behind = 0
            if (bus_object.current_load + ons - offs) > vehicle_capacity:
                left_behind = bus_object.current_load + ons - offs - vehicle_capacity
                ons = ons - left_behind
            
            # if people already left
            if bLeft:
                ons = 0
                
            bus_object.current_load = bus_object.current_load + ons
            bus_object.current_load = bus_object.current_load - offs
            bus_object.total_passengers_served += ons
            log(self.logger, _new_time, f"picking up @ {stop_id}: curr:{current_load},{ons}/smp:{sampled_load},{sampled_ons}", LogType.DEBUG)
            log(self.logger, _new_time, f"Bus {bus_id} @ {stop_id} ons:{ons}, offs:{offs}, curr_load:{bus_object.current_load}", LogType.INFO)

            stop_object.total_passenger_ons += ons
            stop_object.total_passenger_offs += offs
            stop_object.passenger_waiting[key]['boarded'] = ons
            if left_behind > 0:
                log(self.logger, _new_time, f"Bus {bus_id} @ {stop_id} left {left_behind} passengers behind", LogType.ERROR)
                passenger_waiting[key]['left_behind'] = left_behind
                stop_object.passenger_waiting = passenger_waiting