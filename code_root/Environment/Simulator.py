import copy
import time

import pandas as pd
from Environment.enums import BusStatus, BusType, ActionType, EventType
from src.utils import *
import datetime as dt
import logging
import pickle


class Simulator:
    def __init__(
        self,
        starting_state,
        environment_model,
        event_processing_callback,
        starting_event_queue,
        valid_actions,
        config,
        travel_model,
    ) -> None:
        self.state = starting_state
        self.environment_model = environment_model
        self.event_processing_callback = event_processing_callback
        self.event_queue = starting_event_queue
        self.save_metrics = config.get("save_metrics", False)
        self.travel_model = travel_model

        self.start_sim_time = None
        self.starting_num_events = len(starting_event_queue)
        self.num_events_processed = 0
        self.decision_events = 0
        self.config = config
        self.use_intervals = config.get("use_intervals", False)
        self.use_timepoints = config.get("use_timepoints", False)
        self.reallocation = config.get("reallocation", False)
        self.log_name = config["mcts_log_name"]
        self.state_checkpoint = config.get("use_checkpoints", False)

        self.last_visual_log = None
        self.valid_actions = valid_actions

        self.csvlogger = logging.getLogger("csvlogger")
        self.logger = logging.getLogger("debuglogger")

    # Other actors catch up to the event..?
    def run_simulation(self):
        log(self.logger, dt.datetime.now(), "Running simulation (real world time)", LogType.INFO)

        self.start_sim_time = time.time()

        # initialize state
        update_event = None

        while len(self.event_queue) > 0:
            self.update_sim_info()

            chosen_action = None
            if self.valid_actions is not None:
                _valid_actions = self.valid_actions.get_valid_actions(self.state)
            else:
                _valid_actions = None

            if self.config["method"].upper() == "MCTS":
                if self.config["scenario"].upper() == "1A":
                    self.decide_and_take_actions_1A(self.event_queue[0], _valid_actions)
                elif self.config["scenario"].upper() == "1B":
                    self.decide_and_take_actions_1B(self.event_queue[0], _valid_actions)
                elif self.config["scenario"].upper() == "2A":
                    self.decide_and_take_actions_2A(self.event_queue[0], _valid_actions)
            elif self.config["method"].upper() == "BASELINE":
                self.decide_and_take_actions_baseline(self.event_queue[0], _valid_actions)

            # update_event = self.event_queue.pop(0)
            self.event_queue.pop(0)
            if len(self.event_queue) > 0:
                new_events = self.environment_model.update(self.state, self.event_queue[0])
                for event in new_events:
                    self.add_event(event)

            self.state.bus_events = copy.copy(self.event_queue)

            if self.state_checkpoint:
                self.save_state()
                break

            if self.save_metrics:
                self.log_metrics()

            early_end = self.config.get("early_end")
            if early_end and self.state.time >= str_timestamp_to_datetime(early_end):
                break
        # print(f"Done:{dt.datetime.now()}")
        # TODO: Calculate the deadmiles traveled to go back to the main depot (in nestor or MCC)

        self.print_states(csv=True)
        # self.print_res()
        log(self.logger, dt.datetime.now(), "Finished simulation (real world time)", LogType.INFO)

        return self.get_score()

    def get_score(self):
        a = 0
        b = 0
        for bus_id, bus_obj in self.state.buses.items():
            if bus_obj.type == BusType.OVERLOAD:
                tt = 0
                dd = 0
                if "MCC" not in bus_obj.current_stop[0:3]:
                    tt, dd = self.travel_model.get_traveltime_distance_from_stops(
                        bus_obj.current_stop, "MCC5_11", self.state.time
                    )
                a += bus_obj.total_deadkms_moved + bus_obj.partial_deadkms_moved + dd
                b += bus_obj.total_deadsecs_moved + tt

        c = 0
        for stop_id, stop_obj in self.state.stops.items():
            passenger_set_counts = stop_obj.passenger_waiting_dict_list
            for p_set in passenger_set_counts:
                c += p_set["ons"]
        # TODO: For baseline and for future return a,b,c (need to refactor the executors after)
        # return a + b + c
        self.csvlogger.debug(f"{a},{b},{c}")
        return a, b, c

    def decide_and_take_actions_baseline(self, update_event, _valid_actions):
        chosen_action = None
        if update_event is None:
            return

        bus_id = update_event.type_specific_information.get("bus_id")
        if bus_id and self.state.buses[bus_id].type == BusType.OVERLOAD:
            return

        if (update_event.event_type == EventType.DECISION_ALLOCATION_EVENT) and self.reallocation:
            chosen_action = self.event_processing_callback(
                _valid_actions, self.state, action_type=ActionType.OVERLOAD_ALLOCATE
            )
        elif (
            update_event.event_type == EventType.VEHICLE_ARRIVE_AT_STOP
            or update_event.event_type == EventType.PASSENGER_LEFT_BEHIND
        ):
            chosen_action = self.event_processing_callback(
                _valid_actions, self.state, action_type=ActionType.OVERLOAD_DISPATCH
            )
        if chosen_action is None:
            chosen_action = {"type": ActionType.NO_ACTION, "overload_bus": None, "info": None}

        if self.save_metrics:
            self.action_taken_log.debug(f"{self.num_events_processed},{self.state.time},{chosen_action}")

        if chosen_action["type"] != ActionType.NO_ACTION:
            log(self.logger, self.state.time, f"Chosen action:{chosen_action}", LogType.DEBUG)
        new_events, _ = self.environment_model.take_action(self.state, chosen_action, baseline=True)
        for event in new_events:
            self.add_event(event)

    def decide_and_take_actions_1A(self, update_event, _valid_actions):
        if update_event and (update_event.event_type == EventType.DECISION_ALLOCATION_EVENT) and self.reallocation:
            chosen_action = self.event_processing_callback(
                _valid_actions, self.state, action_type=ActionType.OVERLOAD_ALLOCATE
            )
            if chosen_action is None:
                chosen_action = {"type": ActionType.NO_ACTION, "overload_bus": None, "info": None}

            if self.save_metrics:
                self.action_taken_log.debug(f"{self.num_events_processed},{self.state.time},{chosen_action}")
            log(self.logger, self.state.time, f"Chosen action:{chosen_action}", LogType.DEBUG)
            new_events, _ = self.environment_model.take_action(self.state, chosen_action)
            for event in new_events:
                self.add_event(event)
            self.decision_events += 1

            # Only do decision epochs for regular buses
        if (
            update_event
            and self.use_timepoints
            and (
                (update_event.event_type == EventType.VEHICLE_ARRIVE_AT_STOP)
                or (update_event.event_type == EventType.VEHICLE_BREAKDOWN)
                or (update_event.event_type == EventType.PASSENGER_LEFT_BEHIND)
            )
        ):
            bus_id = update_event.type_specific_information.get("bus_id")
            if bus_id and self.state.buses[bus_id].type == BusType.OVERLOAD:
                pass
            elif (
                self.environment_model.travel_model.is_event_a_timepoint(update_event, self.state)
                or update_event.event_type == EventType.PASSENGER_LEFT_BEHIND
                or update_event.event_type == EventType.VEHICLE_BREAKDOWN
            ):
                chosen_action = self.event_processing_callback(
                    _valid_actions, self.state, action_type=ActionType.OVERLOAD_DISPATCH
                )

                if chosen_action is None:
                    chosen_action = {"type": ActionType.NO_ACTION, "overload_bus": None, "info": None}

                if self.save_metrics:
                    self.action_taken_log.debug(f"{self.num_events_processed},{self.state.time},{chosen_action}")

                if self.environment_model.travel_model.is_event_a_timepoint(update_event, self.state):
                    log(self.csvlogger, self.state.time, f"At time point.", LogType.DEBUG)

                log(self.csvlogger, self.state.time, f"Chosen action:{chosen_action}", LogType.DEBUG)

                new_events, _ = self.environment_model.take_action(self.state, chosen_action)
                for event in new_events:
                    self.add_event(event)
                self.decision_events += 1

    # TODO: Check if this is trying to dispatch in the future? Since bus may not have reached the "current_stop" yet.
    # Need to check t_state_change
    def decide_and_take_actions_1B(self, update_event, _valid_actions):
        if update_event and (update_event.event_type == EventType.DECISION_ALLOCATION_EVENT) and self.reallocation:
            chosen_action = self.event_processing_callback(
                _valid_actions, self.state, action_type=ActionType.OVERLOAD_ALLOCATE
            )
            if chosen_action is None:
                chosen_action = {"type": ActionType.NO_ACTION, "overload_bus": None, "info": None}

            if self.save_metrics:
                self.action_taken_log.debug(f"{self.num_events_processed},{self.state.time},{chosen_action}")
            log(self.csvlogger, self.state.time, f"Chosen action:{chosen_action}", LogType.DEBUG)
            new_events, _ = self.environment_model.take_action(self.state, chosen_action)
            for event in new_events:
                self.add_event(event)
            self.decision_events += 1

        if update_event and (
            (update_event.event_type == EventType.VEHICLE_ARRIVE_AT_STOP)
            or (update_event.event_type == EventType.VEHICLE_BREAKDOWN)
            or (update_event.event_type == EventType.PASSENGER_LEFT_BEHIND)
        ):
            bus_id = update_event.type_specific_information.get("bus_id")
            if bus_id and self.state.buses[bus_id].type == BusType.OVERLOAD:
                pass
            elif (
                bus_id
                and self.state.buses[bus_id].last_decision_epoch
                and (
                    (self.state.time - self.state.buses[bus_id].last_decision_epoch)
                    < dt.timedelta(minutes=DECISION_INTERVAL)
                )
                and (update_event.event_type != EventType.PASSENGER_LEFT_BEHIND)
                and (update_event.event_type != EventType.VEHICLE_BREAKDOWN)
            ):
                pass
            elif (
                (update_event.event_type == EventType.VEHICLE_BREAKDOWN)
                or (update_event.event_type == EventType.PASSENGER_LEFT_BEHIND)
                or (
                    self.state.buses[bus_id].last_decision_epoch
                    and (self.state.time - self.state.buses[bus_id].last_decision_epoch)
                    >= dt.timedelta(minutes=DECISION_INTERVAL)
                )
                or (self.state.buses[bus_id].last_decision_epoch is None)
            ):
                chosen_action = self.event_processing_callback(
                    _valid_actions, self.state, action_type=ActionType.OVERLOAD_DISPATCH
                )

                self.state.buses[bus_id].last_decision_epoch = self.state.time
                if chosen_action is None:
                    chosen_action = {"type": ActionType.NO_ACTION, "overload_bus": None, "info": None}

                if self.save_metrics:
                    self.action_taken_log.debug(f"{self.num_events_processed},{self.state.time},{chosen_action}")
                log(self.csvlogger, self.state.time, f"Chosen action:{chosen_action}", LogType.DEBUG)
                new_events, _ = self.environment_model.take_action(self.state, chosen_action)
                for event in new_events:
                    self.add_event(event)
                self.decision_events += 1

    def decide_and_take_actions_2A(self, update_event, _valid_actions):
        if update_event and (update_event.event_type == EventType.DECISION_ALLOCATION_EVENT) and self.reallocation:
            chosen_action = self.event_processing_callback(
                _valid_actions, self.state, action_type=ActionType.OVERLOAD_ALLOCATE
            )
            if chosen_action is None:
                chosen_action = {"type": ActionType.NO_ACTION, "overload_bus": None, "info": None}

            if self.save_metrics:
                self.action_taken_log.debug(f"{self.num_events_processed},{self.state.time},{chosen_action}")
            log(self.csvlogger, self.state.time, f"Chosen action:{chosen_action}", LogType.DEBUG)
            new_events, _ = self.environment_model.take_action(self.state, chosen_action)
            for event in new_events:
                self.add_event(event)
            self.decision_events += 1

        if update_event and (
            (update_event.event_type == EventType.DECISION_DISPATCH_EVENT)
            or (update_event.event_type == EventType.VEHICLE_BREAKDOWN)
            or (update_event.event_type == EventType.PASSENGER_LEFT_BEHIND)
        ):
            bus_id = update_event.type_specific_information.get("bus_id")
            if bus_id and self.state.buses[bus_id].type == BusType.OVERLOAD:
                pass
            else:
                chosen_action = self.event_processing_callback(
                    _valid_actions, self.state, action_type=ActionType.OVERLOAD_DISPATCH
                )

                if chosen_action is None:
                    chosen_action = {"type": ActionType.NO_ACTION, "overload_bus": None, "info": None}

                if self.save_metrics:
                    self.action_taken_log.debug(f"{self.num_events_processed},{self.state.time},{chosen_action}")
                log(self.csvlogger, self.state.time, f"Chosen action:{chosen_action}", LogType.DEBUG)
                new_events, _ = self.environment_model.take_action(self.state, chosen_action)
                for event in new_events:
                    self.add_event(event)
                self.decision_events += 1
                self.state.buses[bus_id].last_decision_epoch = self.state.time

    def update_sim_info(self):
        self.num_events_processed += 1

    def add_event(self, new_event):
        self.event_queue.append(new_event)
        self.event_queue.sort(key=lambda _: _.time, reverse=False)

    def log_metrics(self):
        log_time = self.state.time
        for stop_id, stop_obj in self.state.stops.items():
            if stop_obj.passenger_waiting:
                # self.stop_metrics_log.debug(f"{log_time},{stop_obj}")
                for route_id, v in stop_obj.passenger_waiting.items():
                    for arrival_time, w in v.items():
                        got_on_bus = w["got_on_bus"]
                        remaining = w["remaining"]
                        block = w["block_trip"][0]
                        trip = w["block_trip"][1]
                        ons = w["ons"]
                        offs = w["offs"]
                        output = f"{self.num_events_processed},{stop_id},{arrival_time},{got_on_bus},{remaining},{block},{trip},{ons},{offs},{stop_obj.total_passenger_ons},{stop_obj.total_passenger_offs},{stop_obj.total_passenger_walk_away}"
                        self.stop_metrics_log.debug(f"{log_time},{output}")

        for bus_id, bus_obj in self.state.buses.items():
            status = bus_obj.status
            bus_type = bus_obj.type
            current_load = bus_obj.current_load
            if bus_obj.current_block_trip:
                current_block = bus_obj.current_block_trip[0]
                current_trip = bus_obj.current_block_trip[1]
            else:
                current_block = None
                current_trip = None

            current_stop = bus_obj.current_stop
            time_at_last_stop = bus_obj.time_at_last_stop
            total_passengers_served = bus_obj.total_passengers_served
            capacity = bus_obj.capacity
            deadkms_moved = bus_obj.total_deadkms_moved
            servicekms_moved = bus_obj.total_servicekms_moved

            self.bus_metrics_log.debug(
                f"{self.num_events_processed},{log_time},{bus_id},{status},{bus_type},{capacity},{current_load},{current_block},{current_trip},{current_stop},{time_at_last_stop},{total_passengers_served},{deadkms_moved},{servicekms_moved}"
            )

    def print_states(self, csv=False):
        LOGTYPE = LogType.INFO
        log(self.logger, None, f"Total events processed: {self.num_events_processed}", LOGTYPE)
        log(self.logger, None, f"Total decision epochs: {self.decision_events}", LOGTYPE)
        if not csv:
            for bus_id, bus_obj in self.state.buses.items():
                log(self.csvlogger, None, f"--Bus ID: {bus_id}--", LOGTYPE)
                log(self.csvlogger, None, f"total dwell_time: {bus_obj.dwell_time:.2f} s", LOGTYPE)
                log(
                    self.csvlogger,
                    None,
                    f"aggregate delay_time: {(bus_obj.delay_time / bus_obj.total_stops):.2f} s",
                    LOGTYPE,
                )
                log(self.csvlogger, None, f"total_service_time: {bus_obj.total_service_time:.2f}", LOGTYPE)
                log(self.csvlogger, None, f"total_passengers_served: {bus_obj.total_passengers_served}", LOGTYPE)
                log(self.csvlogger, None, f"total_servicekms_moved: {bus_obj.total_servicekms_moved:.2f} km", LOGTYPE)
                log(self.csvlogger, None, f"total_deadkms_moved: {bus_obj.total_deadkms_moved:.2f} km", LOGTYPE)
                log(self.csvlogger, None, f"current_stop: {bus_obj.current_stop}", LOGTYPE)
                log(self.csvlogger, None, f"status: {bus_obj.status}", LOGTYPE)

            for stop_id, stop_obj in self.state.stops.items():
                # if stop_obj.total_passenger_walk_away > 0:
                log(self.csvlogger, None, f"--Stop ID: {stop_id}--", LOGTYPE)
                log(self.csvlogger, None, f"total_passenger_ons: {stop_obj.total_passenger_ons}", LOGTYPE)
                log(self.csvlogger, None, f"total_passenger_offs: {stop_obj.total_passenger_offs}", LOGTYPE)
                log(self.csvlogger, None, f"total_passenger_walk_away: {stop_obj.total_passenger_walk_away}", LOGTYPE)
        else:
            log(
                self.csvlogger,
                None,
                "bus_id,dwell_time,agg_delay,service_time,total_served,service_kms,current_stop,status,type,total_deadsecs,starting_stop",
                LOGTYPE,
            )
            for bus_id, bus_obj in self.state.buses.items():
                a = bus_id
                b = f"{bus_obj.dwell_time:.2f}"
                c = f"{(bus_obj.delay_time / bus_obj.total_stops):.2f}"
                d = f"{bus_obj.total_service_time:.2f}"
                e = f"{bus_obj.total_passengers_served}"
                f = f"{bus_obj.total_deadkms_moved + bus_obj.partial_deadkms_moved:.2f}"
                g = f"{bus_obj.current_stop}"
                h = f"{bus_obj.status}"
                i = f"{bus_obj.type}"
                j = f"{bus_obj.total_deadsecs_moved:.2f}"  # Deadseconds + trip back to depot(?)
                k = f"{bus_obj.starting_stop}"
                # csvlogger.info(f"{a},{b},{c},{d},{e},{f},{g},{h},{i},{j},{k}")
                log(self.csvlogger, None, f"{a},{b},{c},{d},{e},{f},{g},{h},{i},{j},{k}")

            log(
                self.csvlogger,
                None,
                "stop_id,total_passenger_ons,total_passenger_offs,total_passenger_walk_away",
                LOGTYPE,
            )
            for stop_id, stop_obj in self.state.stops.items():
                a = stop_id
                b = f"{stop_obj.total_passenger_ons}"
                c = f"{stop_obj.total_passenger_offs}"
                # d = f"{stop_obj.total_passenger_walk_away}"
                passenger_set_counts = stop_obj.passenger_waiting_dict_list
                d = 0
                for p_set in passenger_set_counts:
                    d += p_set["ons"]
                # csvlogger.info(f"{a},{b},{c},{d}")
                if float(b) + float(c) + d > 0:
                    log(self.csvlogger, None, f"{a},{b},{c},{d}")

        total_walk_aways = 0
        total_arrivals = 0
        total_boardings = 0
        for stop_id, stop_obj in self.state.stops.items():
            passenger_set_counts = stop_obj.passenger_waiting_dict_list
            stop_walk_aways = 0
            for p_set in passenger_set_counts:
                stop_walk_aways += p_set["ons"]
            total_walk_aways += stop_walk_aways
            total_boardings += stop_obj.total_passenger_ons
            total_arrivals += stop_obj.total_passenger_ons + stop_walk_aways

        log(self.csvlogger, dt.datetime.now(), f"Count of all passengers: {total_arrivals}", LogType.INFO)
        log(self.csvlogger, dt.datetime.now(), f"Count of all passengers who boarded: {total_boardings}", LogType.INFO)
        log(self.csvlogger, dt.datetime.now(), f"Count of all passengers who were left: {total_walk_aways}", LOGTYPE)

    def save_state(self):
        current_time = self.state.time

    def save_visualization(self, event_time, granularity_s=None):
        if not self.last_visual_log:
            self.last_visual_log = event_time

        if granularity_s:
            if event_time - self.last_visual_log < dt.timedelta(seconds=granularity_s):
                return

                # print("Visualizer:", (event_time - self.last_visual_log))
        # self.visual_log.debug(f"time,id,trip_id,last_visited_stop,value,fraction,icon")
        for bus_id, bus_obj in self.state.buses.items():
            if bus_obj.status != BusStatus.IDLE:
                current_trip = ""
                if bus_obj.current_block_trip:
                    current_trip = bus_obj.current_block_trip[1]

                if bus_obj.type == BusType.REGULAR:
                    if bus_obj.status == BusStatus.BROKEN:
                        self.visual_log.debug(
                            f"{event_time},{bus_id},{current_trip},{bus_obj.current_stop},100,{bus_obj.percent_to_next_stop},alert,60"
                        )
                    else:
                        self.visual_log.debug(
                            f"{event_time},{bus_id},{current_trip},{bus_obj.current_stop},{bus_obj.current_load},{bus_obj.percent_to_next_stop},car,35"
                        )
                elif bus_obj.type == BusType.OVERLOAD:
                    self.visual_log.debug(
                        f"{event_time},{bus_id},{current_trip},{bus_obj.current_stop},{bus_obj.current_load},{bus_obj.percent_to_next_stop},car-suv,35"
                    )

        for stop_id, stop_obj in self.state.stops.items():
            pw = stop_obj.passenger_waiting
            ons = 0
            if pw:
                for k, v in pw.items():
                    if len(v) > 0:
                        for j, w in pw[k].items():
                            ons = w["load"]
                            remaining = w["remaining"]
                            if remaining == 0:
                                self.visual_log.debug(f"{event_time},{stop_id},,{stop_id},{ons},,add-person,25")
                            else:
                                self.visual_log.debug(f"{event_time},{stop_id},,{stop_id},{remaining},,add-person,25")

        if granularity_s:
            self.last_visual_log = event_time
