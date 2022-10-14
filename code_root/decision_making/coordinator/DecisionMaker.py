import copy

from src.utils import *
from decision_making.CentralizedMCTS.ModularMCTS import ModularMCTS
from Environment.enums import LogType, EventType
from Environment.DataStructures.Event import Event
import numpy as np
import datetime as dt
import pickle

"""
Combine the two files here:
/media/seconddrive/JP/gits/EMS_DSS/code_root/decision_making/LowLevel/CentralizedMCTS/LowLevelCentMCTSPolicy.py
/media/seconddrive/JP/gits/EMS_DSS/code_root/decision_making/coordinator/LowLevelCoordTest.py

# Here you define the solver which is the ModularMCTS

# The output here is the action (which is obtained by running MCTS solver)
"""

def run_low_level_mcts(arg_dict):
    '''
    arg dict needs:
    current_state,
    event_queue,
    iter_limit,
    allowed_compu_time,
    exploration_constant,
    discount_factor,
    rollout_policy,
    # reward_function,
    # travel_model,
    mdp_environment,
    MCTS_type
    :param arg_dict:
    :return:
    '''

    #(discount_factor=mcts_discount_factor,
    #                    mdp_environment_model=mdp_environment_model,
    #                    rollout_policy=rollout_policy,
    #                    uct_tradeoff=uct_tradeoff,
    #                    iter_limit=iter_limit,
    #                    allowed_computation_time=allowed_computation_time,  # 5 seconds per thread
    #                    logger=mcts_logger)

    solver = ModularMCTS(mdp_environment_model=arg_dict['mdp_environment_model'],
                         discount_factor=arg_dict['discount_factor'],
                         iter_limit=arg_dict['iter_limit'],
                         allowed_computation_time=arg_dict['allowed_computation_time'],
                         rollout_policy=arg_dict['rollout_policy'],
                         logger=arg_dict['logger'])

    res = solver.solve(arg_dict['current_state'], arg_dict['event_queue'])

    return {'region_id': arg_dict['tree_number'],
            'mcts_res': res}

class DecisionMaker:

    def __init__(self,
                 environment_model,
                 travel_model,
                 dispatch_policy,
                 logger,
                 pool_thread_count,
                 mcts_type,
                 discount_factor,
                 mdp_environment_model,
                 rollout_policy,
                 uct_tradeoff,
                 iter_limit,
                 lookahead_horizon_delta_t,
                 allowed_computation_time):
        self.environment_model         = environment_model
        self.travel_model              = travel_model
        self.dispatch_policy           = dispatch_policy
        self.logger                    = logger
        self.event_counter             = 0
        self.pool_thread_count         = pool_thread_count
        self.mcts_type                 = mcts_type

        self.discount_factor           = discount_factor
        self.mdp_environment_model     = mdp_environment_model
        self.rollout_policy            = rollout_policy
        self.uct_tradeoff              = uct_tradeoff
        self.iter_limit                = iter_limit
        self.allowed_computation_time  = allowed_computation_time
        self.lookahead_horizon_delta_t = lookahead_horizon_delta_t

    # Call the MCTS in parallel here

    def event_processing_callback_funct(self, actions, state):
        self.event_counter += 1

        # print(f"DecisionMaker::event_processing_callback_funct({self.event_counter})")
        log(self.logger, state.time, f"Event processing callback", LogType.DEBUG)

        chosen_action = self.process_mcts(state)

        return chosen_action

    def process_mcts(self, state):
        # event_queues = self.get_event_chains(state)
        event_queues = self.load_events(state)
        return self.get_action([state], event_queues)

    # TODO: Do i also modify the states for each new tree?
    def get_action(self, states, event_queues):
        final_action = {}

        if self.pool_thread_count == 0:
            res_dict = []
            inputs = self.get_mcts_inputs(states=states,
                                          event_queues=event_queues,
                                          discount_factor=self.discount_factor,
                                          mdp_environment_model=self.mdp_environment_model,
                                          rollout_policy=self.rollout_policy,
                                          uct_tradeoff=self.uct_tradeoff,
                                          iter_limit=self.iter_limit,
                                          allowed_computation_time=self.allowed_computation_time,
                                          mcts_type=self.mcts_type)

            for input in inputs:
                result = run_low_level_mcts(input)
                res_dict.append(result)

            best_actions = dict()

            for i in range(len(res_dict)):
                results = [_['mcts_res'] for _ in res_dict if _['region_id'] == i]
                actions = [_['action'] for _ in results[0]['scored_actions']]

                all_action_scores = []
                for action in actions:
                    action_scores = []
                    for result in results:
                        action_score = next((_ for _ in result['scored_actions'] if _['action'] == action), None)
                        action_scores.append(action_score['score'])

                    all_action_scores.append({'action': action, 'scores': action_scores})

                avg_action_scores = list()
                for res in all_action_scores:
                    avg_action_scores.append({'action': res['action'],
                                              'avg_score': np.mean(res['scores'])})

                best_actions[i] = max(avg_action_scores, key=lambda _: _['avg_score'])['action']

            # print(f"DecisionMaker scores:{avg_action_scores}")

            for region_id, action_dict in best_actions.items():
                for resp_id, depot_id in action_dict.items():
                    final_action[resp_id] = depot_id

        # print(f"DecisionMaker res_dict:{res_dict[0]}")
        # print(f"DecisionMaker final action:{final_action}")
        # print(f"DecisionMaker event:{res_dict[0]['mcts_res']['tree'].event_at_node}")

        log(self.logger, states[0].time, f"MCTS Decision: {final_action}")
        return final_action

    # Different number of people going to the stops
    # Question: Should travel time be part of the chains too?
    def get_event_chains(self, state):
        return [state.events]

    def get_mcts_inputs(self,
                        states,
                        event_queues,
                        discount_factor,
                        mdp_environment_model,
                        rollout_policy,
                        uct_tradeoff,
                        iter_limit,
                        allowed_computation_time,
                        mcts_type):
        inputs = []

        # Based on how many parallel mcts we want
        for i in range(1):
            input_dict = {}
            input_dict['tree_number'] = i
            input_dict['MCTS_type'] = mcts_type
            input_dict['mdp_environment_model'] = mdp_environment_model
            input_dict['discount_factor'] = discount_factor
            input_dict['iter_limit'] = iter_limit
            input_dict['allowed_computation_time'] = allowed_computation_time
            input_dict['rollout_policy'] = copy.deepcopy(rollout_policy)
            input_dict['event_queue'] = copy.deepcopy(event_queues[i])
            input_dict['current_state'] = copy.deepcopy(states[i])
            input_dict['logger'] = self.logger
            inputs.append(input_dict)

        return inputs

    """
    * Loading same one as real world environment
    * Issue, our events do not include the buses coming (arrival/travel time)
    * Retain remaining people on stops but sample new ones.
        Bus locations and travel time do not matter since they all start from either a stop or a fraction of the way
    to the stop. If they are a fraction of the way away, do not resample travel time. just repopulate events.
    * See if vehicles will break down. If a bus is at a stop, have a chance it will break down. Loop through buses.
    * Loop through events and re-sample loads while adding any remaining passengers.
    * Don't touch travel times and distances
    """
    def load_events(self, state):
        events = copy.copy(state.events)
        # event_file = 'events_1.pkl'
        # saved_events = f'/media/seconddrive/JP/gits/mta_simulator_redo/code_root/scenarios/baseline/data/{event_file}'
        # with open(saved_events, "rb") as f:
        #     events = pickle.load(f)

        # _events = [event for event in events if state.time <= event.time]

        # Rollout lookahead_horizon
        lookahead_horizon = state.time + dt.timedelta(seconds=self.lookahead_horizon_delta_t)
        _events = [event for event in events if state.time <= event.time <= lookahead_horizon]
        # Preventing empty events
        if len(_events) == 0:
            # print(f"No events in the MCTS event queue!")
            _events = [events[0]]

        return [_events]