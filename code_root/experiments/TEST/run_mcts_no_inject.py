import sys

BASE_DIR = "../../../code_root"
sys.path.append(BASE_DIR)

# All dates and times should just be datetime!
from DecisionMaking.Coordinator.DecisionMaker import DecisionMaker
from DecisionMaking.Coordinator.NearestCoordinator import NearestCoordinator
from DecisionMaking.Dispatch.SendNearestDispatchPolicy import SendNearestDispatchPolicy
from DecisionMaking.Dispatch.HeuristicDispatch import HeuristicDispatch
from DecisionMaking.DecisionEnvironmentDynamics import DecisionEnvironmentDynamics
from DecisionMaking.CentralizedMCTS.Rollout import BareMinimumRollout
from Environment.DataStructures.Bus import Bus
from Environment.DataStructures.Event import Event
from Environment.DataStructures.State import State
from Environment.DataStructures.Stop import Stop
from Environment.EmpiricalTravelModelLookup import EmpiricalTravelModelLookup
from Environment.enums import BusStatus, BusType, EventType, MCTSType, LogType
from Environment.EnvironmentModelFast import EnvironmentModelFast
from Environment.Simulator import Simulator
from src.utils import *
import copy
import time
import logging
import argparse
import json
import pickle
import numpy as np
import pandas as pd
import datetime as dt
from pathlib import Path


def load_initial_state(starting_date, bus_plan, trip_plan, config, random_seed=100):
    # print("Loading initial states...")
    active_stops = []

    _starting_date_str = dt.datetime.strptime(starting_date, "%Y%m%d").strftime("%Y-%m-%d")
    Buses = {}
    for bus_id, bus_info in bus_plan.items():
        bus_type = bus_info["service_type"]
        if bus_type == "regular":
            bus_type = BusType.REGULAR
            bus_starting_depot = bus_info["starting_depot"]
        else:
            if bus_id not in config["overload_start_depots"]:
                # print(f"Bus {bus_id} not in config.")
                continue
            if config["overload_start_depots"].get(bus_id) == "":
                # print(f"Bus {bus_id} not in config.")
                continue
            bus_type = BusType.OVERLOAD
            if config.get("overload_start_depots", False):
                bus_starting_depot = config["overload_start_depots"].get(bus_id, bus_info["starting_depot"])
            else:
                bus_starting_depot = bus_info["starting_depot"]

        bus_status = BusStatus.IDLE
        bus_capacity = bus_info["vehicle_capacity"]
        bus_block_trips = np.asarray(bus_info["trips"])

        bus_block_trips = [tuple(bus_block_trip) for bus_block_trip in bus_block_trips]
        for i, bus_block_trip in enumerate(bus_block_trips):
            block_id = bus_block_trip[0]
            trip_id = bus_block_trip[1]
            trip_info = trip_plan[trip_id]
            stop_id_original = trip_info["stop_id_original"]
            # Make all MCC a single stop
            stop_id_original = ["MCC" if "MCC" in stop_id[0:3] else stop_id for stop_id in stop_id_original]

            active_stops.extend(stop_id_original)
            if i == 0:
                st = trip_plan[trip_id]["scheduled_time"]
                st = [str_timestamp_to_datetime(st).time().strftime("%H:%M:%S") for st in st][0]
                # Add when the bus should reach next stop as state change
                t_state_change = str_timestamp_to_datetime(f"{_starting_date_str} {st}")

        if "MCC" in bus_starting_depot[0:3]:
            bus_starting_depot = "MCC"

        bus = Bus(bus_id, bus_type, bus_status, bus_capacity, bus_block_trips)
        bus.current_stop = bus_starting_depot
        bus.starting_stop = bus_starting_depot
        bus.current_load = 0
        bus.t_state_change = t_state_change
        Buses[bus_id] = bus

    Stops = {}
    for active_stop in active_stops:
        stop = Stop(stop_id=active_stop)
        Stops[active_stop] = stop

    # print(f"Added {len(Buses)} buses and {len(Stops)} stops.")
    return Buses, Stops, active_stops


def load_events(REALWORLD_DIR, travel_model, starting_date, Buses, Stops, trip_plan, event_file="", random_seed=100):
    datetime_str = dt.datetime.strptime(starting_date, "%Y%m%d")
    # print("Adding events...")
    np.random.seed(random_seed)

    # Initial events
    # Includes: Trip starts, passenger sampling
    # all active stops that buses will pass
    events = []
    saved_events = f"scenarios/testset/{datetime_str}/{event_file}"

    _starting_date_str = dt.datetime.strptime(starting_date, "%Y%m%d").strftime("%Y-%m-%d")

    active_trips = []
    pbar = Buses.items()
    # if not os.path.exists(saved_events):
    if True:
        for bus_id, bus in pbar:
            if bus.type == BusType.OVERLOAD:
                continue
            blocks_trips = bus.bus_block_trips

            # Start trip (assuming trips are in sequential order)
            block = blocks_trips[0][0]
            trip = blocks_trips[0][1]
            st = trip_plan[trip]["scheduled_time"]
            st = [str_timestamp_to_datetime(st).time().strftime("%H:%M:%S") for st in st][0]
            first_stop_scheduled_time = str_timestamp_to_datetime(f"{_starting_date_str} {st}")
            current_block_trip = bus.bus_block_trips.pop(0)
            current_stop_number = bus.current_stop_number
            current_depot = bus.current_stop

            active_trips.append(current_block_trip[1])

            bus.current_block_trip = current_block_trip
            bus.current_stop_number = current_stop_number

            travel_time, distance = travel_model.get_traveltime_distance_from_depot(
                current_block_trip, current_depot, bus.current_stop_number, first_stop_scheduled_time
            )

            time_to_state_change = first_stop_scheduled_time + dt.timedelta(seconds=travel_time)
            bus.t_state_change = time_to_state_change
            bus.distance_to_next_stop = distance
            bus.status = BusStatus.IN_TRANSIT

            route_id_direction = travel_model.get_route_id_direction(current_block_trip)

            event = Event(
                event_type=EventType.VEHICLE_ARRIVE_AT_STOP,
                time=time_to_state_change,
                type_specific_information={
                    "bus_id": bus_id,
                    "current_block_trip": current_block_trip,
                    "stop": bus.current_stop_number,
                    "stop_id": bus.current_stop,
                    "route_id_direction": route_id_direction,
                },
            )
            events.append(event)

        events.sort(key=lambda x: x.time, reverse=False)

        # with open(saved_events, "wb") as f:
        #     pickle.dump(events, f)
    else:
        print("loading events...")
        with open(saved_events, "rb") as f:
            events = pickle.load(f)

    return events, active_trips


# TODO: Double check if all vehicles are there.
def manually_insert_disruption(events, buses, bus_id, time):
    if bus_id not in list(buses.keys()):
        # raise "Bus does not exist."
        return events

    # If overload bus, don't add
    if buses[bus_id].type == BusType.OVERLOAD:
        return events

    start_time = events[0].time
    end_time = events[-1].time

    # if time < start_time or time > end_time:
    #     raise "Time beyond limits."

    if time < start_time:
        print(f"Chosen time is in the past, {time} < {start_time}")
        return events
        # raise "Chosen time is in the past."

    event = Event(event_type=EventType.VEHICLE_BREAKDOWN, time=time, type_specific_information={"bus_id": bus_id})
    events.append(event)
    events.sort(key=lambda x: x.time, reverse=False)
    return events


# TODO: Fix config json to add CHAIN_DIR
# Only load passenger events which will be visited by buses (to keep it fair)
# passenger_waiting_dict_list should be a 2D list (one list per chain) then just use the chain required.
# first element should be the real world. But what a
def load_passengers_events(Stops, active_stops, REALWORLD_DIR, starting_date_str, chain=None, active_trips=[]):
    print(f"Start passenger data load.")
    start = time.time()
    list_of_stops = []
    all_ons = []
    for c in range(chain):
        _stops = copy.deepcopy(Stops)
        # HACK Switched df for testing
        if not c:
            df = pd.read_parquet(f"{REALWORLD_DIR}/sampled_ons_offs_dict_{starting_date_str}_FIX.parquet")
            print("Using initial chain.")
        else:
            df = pd.read_parquet(
                f"{BASE_DIR}/scenarios/DISRUPTION_CHAINS/{starting_date_str}/chains/ons_offs_dict_chain_{starting_date_str}_{chain - 1}.parquet"
            )
            print("Using chains")
        df["stop_id"] = np.where(df["stop_id"].str.startswith("MCC"), "MCC", df["stop_id"])
        df = df[df["stop_id"].isin(active_stops)]
        df = df[df["trip_id"].isin(active_trips)]

        # df = df[(df["ons"] > 0) | (df["offs"] > 0)]
        # Use np.where to apply the conversion condition
        # TODO Problem is in all chains, not all steps are present, need to get superset.
        # FIX is to just check if chain exists on handle passenger arrival()..
        print("MCC" in df.stop_id.unique().tolist())
        df = df.drop(columns=["sampled_loads", "next_load"], errors="ignore")
        all_ons.append(df["ons"].sum())
        for stop_id, stop_df in df.groupby("stop_id"):
            if stop_id not in active_stops:
                continue
            arrival_input = stop_df.to_dict(orient="records")
            _stops[stop_id].passenger_waiting_dict_list = arrival_input
        list_of_stops.append(_stops)
    elapsed = time.time() - start
    print(f"Done loading in {elapsed:0f} seconds")
    max_ons = max(all_ons)
    return list_of_stops, max_ons


def manually_insert_allocation_events(bus_arrival_events, starting_date, buses, trip_plan, intervals=15):
    earliest_datetime = starting_date + dt.timedelta(days=2)
    latest_datetime = starting_date
    for bus_id, bus_obj in buses.items():
        if bus_obj.type == BusType.OVERLOAD:
            continue

        bus_block_trips = bus_obj.bus_block_trips + [bus_obj.current_block_trip]
        for bbt in bus_block_trips:
            trip = bbt[1]
            plan = trip_plan[trip]
            schedule = plan["scheduled_time"]
            first_stop = str_timestamp_to_datetime(schedule[0])
            last_stop = str_timestamp_to_datetime(schedule[-1])
            if first_stop <= earliest_datetime:
                earliest_datetime = first_stop
            if last_stop >= latest_datetime:
                latest_datetime = last_stop

    # Use actual times and dont round down/up
    # earliest_datetime = earliest_datetime - dt.timedelta(minutes=30)
    if latest_datetime.hour < 23:
        # latest_datetime = latest_datetime.replace(hour=latest_datetime.hour + 1, minute=0, second=0, microsecond=0)
        latest_datetime = latest_datetime + dt.timedelta(minutes=15)
    else:
        latest_datetime = latest_datetime.replace(hour=latest_datetime.hour, minute=59, second=0, microsecond=0)

    datetime_range = pd.date_range(earliest_datetime, latest_datetime, freq=f"{intervals}min")
    # datetime_range = np.arange(earliest_datetime, latest_datetime, np.timedelta64(intervals, 'm'))
    for _dt in datetime_range:
        event = Event(
            event_type=EventType.DECISION_ALLOCATION_EVENT, time=_dt, type_specific_information={"bus_id": None}
        )
        bus_arrival_events.append(event)

    bus_arrival_events.sort(key=lambda x: x.time, reverse=False)
    return bus_arrival_events


# Almost the same as the main function.
def run_simulation(config, chain=None):
    # logger = None
    logger = logging.getLogger("debuglogger")

    vehicle_count = config["vehicle_count"]
    starting_date_str = config["starting_date_str"]
    REALWORLD_DIR = f'{BASE_DIR}/scenarios/{config["real_world_dir"]}/{starting_date_str}'
    trip_plan_path = (
        f'{BASE_DIR}/scenarios/{config["real_world_dir"]}/{starting_date_str}/trip_plan_{starting_date_str}.json'
    )

    with open(trip_plan_path) as f:
        trip_plan = json.load(f)

    log(logger, dt.datetime.now(), json.dumps(config), LogType.INFO)

    if vehicle_count != "":
        bus_plan_path = f"{REALWORLD_DIR}/vehicle_plan_{starting_date_str}_{vehicle_count}.json"
    else:
        bus_plan_path = f"{REALWORLD_DIR}/vehicle_plan_{starting_date_str}.json"

    with open(bus_plan_path) as f:
        bus_plan = json.load(f)

    LOOKUP_DIR = f"{BASE_DIR}/scenarios"
    start_time = time.time()
    travel_model = EmpiricalTravelModelLookup(LOOKUP_DIR, starting_date_str, config=config, logger=None)
    elapsed_time = time.time() - start_time
    print(f"Lookup loading time: {elapsed_time:.2f} s")
    dispatch_policy = SendNearestDispatchPolicy(travel_model)  # RandomDispatch(travel_model)

    # TODO: Move to environment model once i know it works
    valid_actions = None

    starting_date = dt.datetime.strptime(starting_date_str, "%Y%m%d")

    Buses, Stops, active_stops = load_initial_state(starting_date_str, bus_plan, trip_plan, config, random_seed=100)
    # print(f"Count buses: {len(Buses)}")

    bus_arrival_events, active_trips = load_events(
        REALWORLD_DIR, travel_model, starting_date_str, Buses, Stops, trip_plan
    )

    stop_chains, max_ons = load_passengers_events(
        Stops, active_stops, REALWORLD_DIR, starting_date_str, chain=chain, active_trips=active_trips
    )
    # print(dt.datetime.now())

    bus_arrival_events = bus_arrival_events

    # Injecting incident
    breakdowns = config.get("breakdowns", False)
    if breakdowns:
        # log(logger, dt.datetime.now(), f"breakdowns are provided: {breakdowns}.", LogType.INFO)
        for bus_id, breakdown_datetime_str in breakdowns.items():
            bus_arrival_events = manually_insert_disruption(
                bus_arrival_events, buses=Buses, bus_id=bus_id, time=str_timestamp_to_datetime(breakdown_datetime_str)
            )
    else:
        print(f"Breakdowns: chain {chain} has no breakdowns: {breakdowns}")
        # raise "No breakdowns key present, even with val: False"

    bus_arrival_events.sort(key=lambda x: x.time, reverse=False)

    # HACK because of my weird simulation event pop, duplicate the first event
    bus_arrival_events.insert(0, bus_arrival_events[0])

    starting_state = copy.deepcopy(
        State(stops=stop_chains[0], buses=Buses, bus_events=bus_arrival_events, time=bus_arrival_events[0].time)
    )

    starting_state.stop_chains = stop_chains

    # Adding interval events
    if config["reallocation"]:
        before_count = len(bus_arrival_events)
        bus_arrival_events = manually_insert_allocation_events(
            bus_arrival_events, starting_date, Buses, trip_plan, intervals=60
        )
        after_count = len(bus_arrival_events)

        log(logger, dt.datetime.now(), f"Initial interval decision events: {after_count - before_count}", LogType.INFO)

    rollout_policy = BareMinimumRollout(
        rollout_horizon_delta_t=config["rollout_horizon_delta_t"], dispatch_policy=dispatch_policy
    )

    mcts_discount_factor = config["mcts_discount_factor"]
    lookahead_horizon_delta_t = config["lookahead_horizon_delta_t"]
    uct_tradeoff = config["uct_tradeoff"]
    # uct_tradeoff = max_ons
    pool_thread_count = config["pool_thread_count"]
    iter_limit = config["iter_limit"]
    allowed_computation_time = config["allowed_computation_time"]
    mcts_type = MCTSType.MODULAR_MCTS

    heuristic_dispatch = HeuristicDispatch(travel_model)
    mdp_environment_model = DecisionEnvironmentDynamics(
        travel_model, dispatch_policy=heuristic_dispatch, config=config
    )

    sim_environment = EnvironmentModelFast(travel_model, config)
    if config["method"] == "MCTS":
        decision_maker = DecisionMaker(
            environment_model=sim_environment,
            travel_model=travel_model,
            dispatch_policy=None,
            pool_thread_count=pool_thread_count,
            mcts_type=mcts_type,
            discount_factor=mcts_discount_factor,
            mdp_environment_model=mdp_environment_model,
            rollout_policy=rollout_policy,
            uct_tradeoff=uct_tradeoff,
            iter_limit=iter_limit,
            lookahead_horizon_delta_t=lookahead_horizon_delta_t,
            allowed_computation_time=allowed_computation_time,  # 5 seconds per thread
            starting_date=starting_date_str,
            oracle=config["oracle"],
            base_dir=f"{BASE_DIR}/scenarios",
            config=config,
        )
    elif config["method"] == "baseline":
        decision_maker = NearestCoordinator(travel_model=travel_model, dispatch_policy=dispatch_policy, config=config)

    simulator = Simulator(
        starting_event_queue=copy.deepcopy(bus_arrival_events),
        starting_state=starting_state,
        environment_model=sim_environment,
        event_processing_callback=decision_maker.event_processing_callback_funct,
        valid_actions=valid_actions,
        config=config,
        travel_model=travel_model,
    )
    start_time = time.time()
    score = simulator.run_simulation()
    print(score)
    elapsed = time.time() - start_time
    csvlogger.info(f"Simulator run time: {elapsed:.2f} s")
    print(f"Simulator run time: {elapsed:.2f} s")
    return score


if __name__ == "__main__":
    # Create the ArgumentParser object
    parser = argparse.ArgumentParser(description="Stationing and dispatch executor.")
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_level", type=str, default="DEBUG")
    parser.add_argument("-c", "--config", type=str, default="configs/test")
    args = parser.parse_args()
    args = namespace_to_dict(args)

    config_path = f'{args["config"]}.json'
    with open(config_path) as f:
        config = json.load(f)

    _dir = "."
    filename = config["mcts_log_name"]
    exp_log_path = f"{_dir}/logs/{config['starting_date_str']}_{filename}"
    exp_res_path = f"{_dir}/results/{config['starting_date_str']}_{filename}"
    Path(exp_log_path).mkdir(parents=True, exist_ok=True)
    Path(exp_res_path).mkdir(parents=True, exist_ok=True)

    logFormatter = logging.Formatter("[%(asctime)s][%(levelname)-.1s] %(message)s", "%m-%d %H:%M:%S")
    logger = logging.getLogger("debuglogger")
    logger.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler(f"{exp_log_path}/stream.log", mode="w")
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(logging.DEBUG)

    csvlogger = logging.getLogger("csvlogger")
    csvlogger.setLevel(logging.DEBUG)
    res_file = f"{exp_res_path}/results.csv"
    csvFormatter = logging.Formatter("%(message)s")
    csvHandler = logging.FileHandler(f"{res_file}", mode="w")
    csvHandler.setFormatter(csvFormatter)
    csvHandler.setLevel(logging.DEBUG)
    csvlogger.addHandler(csvHandler)
    # csvlogger.addHandler(logging.NullHandler())

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(logFormatter)
    streamHandler.setLevel(logging.DEBUG)

    # logger.addHandler(fileHandler)
    logger.addHandler(logging.NullHandler())
    # logger.addHandler(streamHandler)

    logger.debug("Starting process.")
    logger.debug(f"pool_thread_count: {config.get('pool_thread_count', 0)}")
    run_simulation(config, chain=config.get("pool_thread_count", 0))

    if config.get("send_mail", False):
        emailer(config_path)
