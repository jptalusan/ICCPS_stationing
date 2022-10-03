import copy
import datetime as dt
from Environment.enums import BusStatus, BusType, EventType
from Environment.DataStructures.Event import Event
from src.utils import *
import math

class BusDynamics:
    
    def __init__(self, travel_model, logger) -> None:
        self.travel_model = travel_model
        self.logger = logger
        self.new_events = []
    
    def update_bus(self, curr_event, bus_id, _new_time, full_state):
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
                new_resp_time, update_status = self.broken_update(bus_id, curr_event, curr_bus_time, _new_time, full_state)
            elif resp_status == BusStatus.IDLE:
                new_resp_time, update_status = self.idle_update(bus_id, curr_event, curr_bus_time, _new_time, full_state)
            elif resp_status == BusStatus.IN_TRANSIT:
                new_resp_time, update_status = self.transit_update(bus_id, curr_event, curr_bus_time, _new_time, full_state)
            elif resp_status == BusStatus.ALLOCATION:
                new_resp_time, update_status = self.allocation_update(bus_id, curr_event, curr_bus_time, _new_time, full_state)
                
            curr_bus_time = new_resp_time
        
        # log(self.logger, curr_bus_time, f"Before if: {curr_bus_time}, {_new_time}")
        if curr_bus_time == _new_time:
            resp_status = full_state.buses[bus_id].status

            new_resp_time = None
            
            if resp_status == BusStatus.BROKEN:
                new_resp_time, update_status = self.broken_update(bus_id, curr_event, curr_bus_time, _new_time, full_state)
            elif resp_status == BusStatus.IDLE:
                new_resp_time, update_status = self.idle_update(bus_id, curr_event, curr_bus_time, _new_time, full_state)
            elif resp_status == BusStatus.IN_TRANSIT:
                new_resp_time, update_status = self.transit_update(bus_id, curr_event, curr_bus_time, _new_time, full_state)
            elif resp_status == BusStatus.ALLOCATION:
                new_resp_time, update_status = self.allocation_update(bus_id, curr_event, curr_bus_time, _new_time, full_state)
                
            curr_bus_time = new_resp_time
            
        return self.new_events
    
    def broken_update(self, bus_id, curr_event, curr_bus_time, _new_time, full_state):
        full_state.buses[bus_id].status = BusStatus.BROKEN
        
        return _new_time, False

    # TODO: Add dead kms
    def allocation_update(self, bus_id, curr_event, curr_bus_time, _new_time, full_state):
        if _new_time >= full_state.buses[bus_id].t_state_change:
            full_state.buses[bus_id].status = BusStatus.IDLE
            time_of_reallocation = full_state.buses[bus_id].t_state_change
            log(self.logger, _new_time, f"Reallocated Bus {bus_id} to {full_state.buses[bus_id].current_stop}")
            return time_of_reallocation, False
        
        # Interpolate
        else:
            if full_state.buses[bus_id].time_at_last_stop:
                journey_fraction = (_new_time - full_state.buses[bus_id].time_at_last_stop) / (full_state.buses[bus_id].t_state_change - full_state.buses[bus_id].time_at_last_stop)
            else:
                journey_fraction = 0.0
            log(self.logger, _new_time, f"Reallocating Bus {bus_id}: {journey_fraction*100:.2f}% to {full_state.buses[bus_id].current_stop_number}")
            full_state.buses[bus_id].percent_to_next_stop = journey_fraction

            return _new_time, False
    
    def idle_update(self, bus_id, curr_event, curr_bus_time, _new_time, full_state):
        # self.logger.debug("Enter idle_update()")
        '''
        In this case, there is nothing to update for the bus' state, since it is waiting (between trips, before trips).
        '''
        # Activate bus if idle and has trips
        if _new_time >= full_state.buses[bus_id].t_state_change:
            time_of_activation = full_state.buses[bus_id].t_state_change
            # Move trips
            if len(full_state.buses[bus_id].bus_block_trips) > 0:
                full_state.buses[bus_id].status = BusStatus.IN_TRANSIT
                current_block_trip = full_state.buses[bus_id].bus_block_trips.pop(0)
                full_state.buses[bus_id].current_block_trip = current_block_trip
                bus_block_trips = full_state.buses[bus_id].bus_block_trips
                # Assuming idle buses are start a depot
                current_depot = full_state.buses[bus_id].current_stop
                
                travel_time = self.travel_model.get_travel_time_from_depot(current_block_trip, current_depot, full_state.buses[bus_id].current_stop_number, _new_time)
                time_to_state_change = time_of_activation + dt.timedelta(seconds=travel_time)
                # print(f"id:{bus_id},tt:{travel_time},{time_of_activation},{time_to_state_change}")
                full_state.buses[bus_id].t_state_change = time_to_state_change
                full_state.buses[bus_id].time_at_last_stop = time_of_activation
                
                # For distance
                next_stop_id = self.travel_model.get_stop_id_at_number(current_block_trip, full_state.buses[bus_id].current_stop_number)
                full_state.buses[bus_id].distance_to_next_stop = self.travel_model.get_distance(current_depot, next_stop_id, _new_time)
                
                event = Event(event_type=EventType.VEHICLE_ARRIVE_AT_STOP, 
                              time=time_to_state_change,
                              type_specific_information={'bus_id': bus_id, 
                                                         'current_block_trip': current_block_trip,
                                                         'stop':full_state.buses[bus_id].current_stop_number})
                self.new_events.append(event)
                return time_of_activation, True
            # no more trips left
            else:
                pass
        # No trips in the future, if they are idle and no trips in the future, set as overload
        else:
            full_state.buses[bus_id].status = BusStatus.IDLE
            # full_state.buses[bus_id].type = BusType.OVERLOAD
            pass
        
        return _new_time, False
    
    # If curr_event == BROKEN and it was in transit, set to broken!
    def transit_update(self, bus_id, curr_event, curr_bus_time, _new_time, full_state):
        '''
        In this case, we update the position of the bus
        update current stop number <- next_stop_number
        update next_stop_number <- new next stop
        '''
        # log(self.logger, _new_time, f"bus time:{curr_bus_time}, bus state change:{full_state.buses[bus_id].t_state_change}")
        
        # print(f"transit_update: {curr_event}, {_new_time}, {full_state.buses[bus_id].t_state_change}")
        if curr_event.event_type == EventType.VEHICLE_BREAKDOWN:
            type_specific_information = curr_event.type_specific_information
            event_bus_id = type_specific_information['bus_id']
            current_block_trip = full_state.buses[event_bus_id].current_block_trip
            if event_bus_id == bus_id:
                print(f"IN_TRANSIT -> BREAKDOWN! bus {bus_id}")
                log(self.logger, _new_time, f"Bus {bus_id} on trip: {current_block_trip[1]} scheduled for broke down.", LogType.ERROR)
                print(curr_event)
                full_state.buses[bus_id].status = BusStatus.BROKEN
                full_state.buses[bus_id].t_state_change = curr_bus_time
                return curr_bus_time, False
        
        # Calculate travel time to next stop and update t_state_change
        if _new_time >= full_state.buses[bus_id].t_state_change:
            time_of_arrival        = full_state.buses[bus_id].t_state_change
            current_block_trip     = full_state.buses[bus_id].current_block_trip
            bus_block_trips        = full_state.buses[bus_id].bus_block_trips
            current_stop_number    = full_state.buses[bus_id].current_stop_number
            scheduled_arrival_time = self.travel_model.get_scheduled_arrival_time(current_block_trip, current_stop_number)
            current_stop_id        = self.travel_model.get_stop_id_at_number(current_block_trip, current_stop_number)
            last_stop_number       = self.travel_model.get_last_stop_number_on_trip(current_block_trip)
            
            log(self.logger, _new_time, f"Bus {bus_id} on trip: {current_block_trip[1]} scheduled for {scheduled_arrival_time} arrives at stop: {current_stop_id}", LogType.INFO)
            # Bus running time
            if full_state.buses[bus_id].time_at_last_stop:
                full_state.buses[bus_id].total_service_time += (_new_time - full_state.buses[bus_id].time_at_last_stop).total_seconds()
                
            full_state.buses[bus_id].time_at_last_stop = time_of_arrival
            full_state.buses[bus_id].current_stop = current_stop_id
            
            # If valid stop
            if full_state.buses[bus_id].current_stop_number >= 0:
                self.pickup_passengers(_new_time, bus_id, current_stop_id, full_state)
                log(self.logger, _new_time, f"Stop:{current_stop_id}:{full_state.stops[current_stop_id].passenger_waiting}")
            
            # No next stop but maybe has next trips? (will check in idle_update)
            if current_stop_number == last_stop_number:
                full_state.buses[bus_id].current_stop_number = 0
                full_state.buses[bus_id].status = BusStatus.IDLE

            # Going to next stop
            else:
                full_state.buses[bus_id].current_stop_number = current_stop_number + 1
                travel_time            = self.travel_model.get_travel_time(current_block_trip, current_stop_number, _new_time)
                scheduled_arrival_time = self.travel_model.get_scheduled_arrival_time(current_block_trip, full_state.buses[bus_id].current_stop_number)
                
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
                
                # For distance
                full_state.buses[bus_id].total_servicekms_moved += full_state.buses[bus_id].distance_to_next_stop
                next_stop_id = self.travel_model.get_stop_id_at_number(current_block_trip, full_state.buses[bus_id].current_stop_number)
                full_state.buses[bus_id].distance_to_next_stop = self.travel_model.get_distance(current_stop_id, next_stop_id, _new_time)
                
                event = Event(event_type=EventType.VEHICLE_ARRIVE_AT_STOP, 
                              time=time_to_state_change,
                              type_specific_information={'bus_id': bus_id, 
                                                         'current_block_trip': current_block_trip,
                                                         'stop':full_state.buses[bus_id].current_stop_number})
                self.new_events.append(event)
            
            return time_of_arrival, False
        
        # Interpolate
        else:
            if full_state.buses[bus_id].time_at_last_stop:
                journey_fraction = (_new_time - full_state.buses[bus_id].time_at_last_stop) / (full_state.buses[bus_id].t_state_change - full_state.buses[bus_id].time_at_last_stop)
            else:
                journey_fraction = 0.0
            # log(self.logger, _new_time, f"Bus {bus_id}: {journey_fraction*100:.2f}% to {full_state.buses[bus_id].current_stop_number}")
            full_state.buses[bus_id].percent_to_next_stop = journey_fraction
            
            distance_fraction = (journey_fraction * full_state.buses[bus_id].distance_to_next_stop)
            log(self.logger, _new_time, f"Bus {bus_id}: {journey_fraction*100:.2f}% to {full_state.buses[bus_id].current_stop_number}")
            log(self.logger, _new_time, f"Bus {bus_id}: {journey_fraction*100:.2f}% to {distance_fraction:.2f}/{full_state.buses[bus_id].distance_to_next_stop:.2f} kms total")

            return _new_time, False
            
    # TODO: Handle zero_load_at_trip_end
    # TODO: add loads processing.
    def pickup_passengers(self, _new_time, bus_id, stop_id, full_state):
        # print("pickup_passengers:", _new_time, bus_id, stop_id)
        bus_object = full_state.buses[bus_id]
        stop_object = full_state.stops[stop_id]
        
        vehicle_capacity    = bus_object.capacity
        current_block_trip  = bus_object.current_block_trip
        current_stop_number = bus_object.current_stop_number
        current_load        = bus_object.current_load
        
        passenger_waiting = stop_object.passenger_waiting
        
        route_id_dir      = self.travel_model.get_route_id_dir_for_trip(current_block_trip)
        last_stop_in_trip = self.travel_model.get_last_stop_number_on_trip(current_block_trip)
        
        # TODO: Not sure if i should sum up offs too.
        # HACKK: Some hacky solutions to handling overflow buses,
        # NOTE: i'm still confused how to get ons if curr load > sampled load
        ons = 0
        offs = 0
        remaining = 0
        sampled_load = 0
        # log(self.logger, _new_time, passenger_waiting)
        if not passenger_waiting:
            return
        if route_id_dir in passenger_waiting:
            for passenger_arrival_time, sampled_data in passenger_waiting[route_id_dir].items():
                assert passenger_arrival_time <= _new_time
                remaining = sampled_data['remaining']
                sampled_load  = sampled_data['load']
                log(self.logger, _new_time, f"BusDynamics1: Bus {bus_id} @ {stop_id}: Sampled/Curr Load: {sampled_load:.0f}/{bus_object.current_load:.0f}", LogType.INFO)
        
                if remaining > 0:
                    sampled_ons  = remaining
                    # Where will i get the value for this?
                    sampled_offs = 0
                    if current_load > sampled_load:
                        percent_offs = (current_load - sampled_load) / current_load
                        sampled_offs = math.floor(percent_offs * current_load)
                else:
                    # Compute ons based on loads
                    
                    if current_load > sampled_load:
                        percent_offs = (current_load - sampled_load) / current_load
                        sampled_offs = math.floor(percent_offs * current_load)
                        sampled_ons  = 0
                    elif current_load < sampled_load:
                        sampled_offs = 0
                        sampled_ons  = sampled_load - current_load
                    else: # current load and sampled load are equal
                        sampled_offs = 0
                        sampled_ons  = 0

                ons  += sampled_ons
                offs += sampled_offs
                break

        # Passenger load, board, alight computations
        offs = min(bus_object.current_load, offs)
        
        if (bus_object.current_load + ons - offs) > vehicle_capacity:
            remaining = bus_object.current_load + ons - offs - vehicle_capacity
        else:
            remaining = 0
            
        if current_stop_number == 0:
            offs = 0
        elif current_stop_number == last_stop_in_trip:
            offs = bus_object.current_load
            ons = 0
            remaining = 0
        
        # log(self.logger, _new_time, f"{ons},{offs},{remaining}", LogType.ERROR)
        bus_object.current_load = bus_object.current_load + ons - offs - remaining
        
        # Delete passenger_waiting
        if remaining == 0:
            passenger_waiting[route_id_dir] = {}
        else:
            passenger_waiting[route_id_dir] = {passenger_arrival_time: {'load':ons, 'remaining':remaining, 'block_trip': current_block_trip}}
            
        log(self.logger, _new_time, f"BusDynamics: Bus {bus_id} @ {stop_id}: on:{ons:.0f}, offs:{offs:.0f}, remain:{remaining:.0f}. Sampled/Curr Load: {sampled_load:.0f}/{bus_object.current_load:.0f}", LogType.INFO)
        # log(self.logger, _new_time, f"BusDynamics: Bus {bus_id} @ {current_block_trip}: {route_id_dir}, {passenger_waiting}")
        stop_object.passenger_waiting[route_id_dir] = passenger_waiting[route_id_dir]
        # stop_object.passenger_waiting = passenger_waiting
        stop_object.total_passenger_ons += ons
        stop_object.total_passenger_offs += offs
        bus_object.total_passengers_served += ons
        bus_object.total_stops += 1