import copy
import pickle
import time
import numpy as np
import datetime as dt
from multiprocessing import Pool
from Environment.DataStructures.Event import Event
from Environment.enums import LogType, EventType
from decision_making.CentralizedMCTS.ModularMCTS import ModularMCTS
from src.utils import *

"""
Combine the two files here:
/media/seconddrive/JP/gits/EMS_DSS/code_root/decision_making/LowLevel/CentralizedMCTS/LowLevelCentMCTSPolicy.py
/media/seconddrive/JP/gits/EMS_DSS/code_root/decision_making/coordinator/LowLevelCoordTest.py

# Here you define the solver which is the ModularMCTS

# The output here is the action (which is obtained by running MCTS solver)
"""

def run_low_level_mcts(arg_dict):
    """
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
    """

    solver = ModularMCTS(mdp_environment_model=arg_dict['mdp_environment_model'],
                         discount_factor=arg_dict['discount_factor'],
                         iter_limit=arg_dict['iter_limit'],
                         allowed_computation_time=arg_dict['allowed_computation_time'],
                         rollout_policy=arg_dict['rollout_policy'],
                        #  logger=arg_dict['logger'],
                         exploit_explore_tradoff_param=arg_dict['exploit_explore_tradoff_param'])

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
                 allowed_computation_time,
                 event_chain_dir):
        self.environment_model = environment_model
        self.travel_model = travel_model
        self.dispatch_policy = dispatch_policy
        self.logger = logger
        self.event_counter = 0
        self.pool_thread_count = pool_thread_count
        self.mcts_type = mcts_type

        self.discount_factor = discount_factor
        self.mdp_environment_model = mdp_environment_model
        self.rollout_policy = rollout_policy
        self.uct_tradeoff = uct_tradeoff
        self.iter_limit = iter_limit
        self.allowed_computation_time  = allowed_computation_time
        self.lookahead_horizon_delta_t = lookahead_horizon_delta_t
        
        self.event_chain_dir = event_chain_dir

    # Call the MCTS in parallel here

    def event_processing_callback_funct(self, actions, state):
        self.event_counter += 1

        chosen_action = self.process_mcts(state)

        if chosen_action is None:
            return None
        return chosen_action

    def process_mcts(self, state):
        event_queues = self.get_event_chains(state)
        # event_queues = self.load_events(state)
        if len(event_queues[0]) <= 0:
            print("No event available...")
            # raise "No event available. should not happen?"
            return None

        result = self.get_action([state], event_queues)
        return result

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

                # We want the actions which result in the least passengers left behind
                best_actions[i] = max(avg_action_scores, key=lambda _: _['avg_score'])['action']

            # print(f"DecisionMaker scores:{avg_action_scores}")

            for region_id, action_dict in best_actions.items():
                for resp_id, action_id in action_dict.items():
                    final_action[resp_id] = action_id
        
        #TODO: Add root parallelization here
        else:

            start_pool_time = time.time()
            with Pool(processes=self.pool_thread_count) as pool:

                pool_creation_time = time.time() - start_pool_time

                inputs = self.get_mcts_inputs(states=states,
                                              event_queues=event_queues,
                                              discount_factor=self.discount_factor,
                                              mdp_environment_model=self.mdp_environment_model,
                                              rollout_policy=self.rollout_policy,
                                              uct_tradeoff=self.uct_tradeoff,
                                              iter_limit=self.iter_limit,
                                              allowed_computation_time=self.allowed_computation_time,
                                              mcts_type=self.mcts_type)

                # run_start_ = time.time()
                res_dict = pool.map(run_low_level_mcts, inputs)

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

                # We want the actions which result in the least passengers left behind
                best_actions[i] = max(avg_action_scores, key=lambda _: _['avg_score'])['action']

            # print(f"DecisionMaker scores:{avg_action_scores}")

            for region_id, action_dict in best_actions.items():
                for resp_id, action_id in action_dict.items():
                    final_action[resp_id] = action_id
                
        print(f"Event counter: {self.event_counter}")
        print(f"DecisionMaker res_dict:{res_dict[0]}")
        print(f"DecisionMaker final action:{final_action}")
        print(f"DecisionMaker event:{res_dict[0]['mcts_res']['tree'].event_at_node}")
        print()
        return final_action

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
        # QUESTION: Copy? deepcopy? plain?
        for i in range(1):
            input_dict = {}
            input_dict['tree_number'] = i
            input_dict['MCTS_type'] = mcts_type
            input_dict['mdp_environment_model'] = mdp_environment_model
            input_dict['discount_factor'] = discount_factor
            input_dict['iter_limit'] = iter_limit
            input_dict['exploit_explore_tradoff_param'] = uct_tradeoff
            input_dict['allowed_computation_time'] = allowed_computation_time
            input_dict['rollout_policy'] = rollout_policy
            input_dict['event_queue'] = copy.deepcopy(event_queues[i])
            input_dict['current_state'] = copy.deepcopy(states[i])
            # input_dict['logger'] = self.logger
            inputs.append(input_dict)

        return inputs

    """
    TODO: A LOT
    * Loading same one as real world environment
    * Issue, our events do not include the buses coming (arrival/travel time)
    * Retain remaining people on stops but sample new ones.
        Bus locations and travel time do not matter since they all start from either a stop or a fraction of the way
    to the stop. If they are a fraction of the way away, do not resample travel time. just repopulate events.
    * See if vehicles will break down. If a bus is at a stop, have a chance it will break down. Loop through buses.
    * Loop through events and re-sample loads while adding any remaining passengers.
    * Don't touch travel times and distances
    *
    * OR: just create chains from data_generation/generate_day_trips.ipynb for ons/offs
    * AND: just generate new probabilities for breakdown based on bus location
    """
    def load_events(self, state):
        events = copy.copy(state.events)
        
        if state.time.time() == dt.time(0, 0, 0):
            start_time = events[0].time
        else:
            start_time = state.time
            
        # Rollout lookahead_horizon
        if self.lookahead_horizon_delta_t:
            lookahead_horizon = start_time + dt.timedelta(seconds=self.lookahead_horizon_delta_t)
            _events = [event for event in events if start_time <= event.time <= lookahead_horizon]
        else:
            _events = [event for event in events if start_time <= event.time]
            
        # if len(_events) <= 0:
        #     _events = events[0]
            
        return [_events]

    # Generate processed chains using generate_chains_pickles.ipynb
    def get_event_chains(self, state, chain_count=1):
        chain_dir = f'scenarios/baseline/chains/{self.event_chain_dir}'
        event_chains = []
        state_events = copy.copy(state.events)
        
        for chain in range(chain_count):
            fp = f'{chain_dir}/ons_offs_dict_chain_{chain + 1}_processed.pkl'
            with open(fp, 'rb') as handle:
                sampled_ons_offs_dict = pickle.load(handle)
                
            original = [pe for pe 
                        in state_events 
                        if (pe.event_type != EventType.PASSENGER_ARRIVE_STOP) 
                        and (pe.event_type != EventType.PASSENGER_LEAVE_STOP)]
            
            chain = [pe for pe 
                     in sampled_ons_offs_dict 
                     if (pe.event_type == EventType.PASSENGER_ARRIVE_STOP) 
                     or (pe.event_type == EventType.PASSENGER_LEAVE_STOP)]
            new_chain = original + chain
            new_chain.sort(key=lambda x: x.time, reverse=False)

            if state.time.time() == dt.time(0, 0, 0):
                start_time = new_chain[0].time
            else:
                start_time = state.time
                
            # Rollout lookahead_horizon
            if self.lookahead_horizon_delta_t:
                lookahead_horizon = start_time + dt.timedelta(seconds=self.lookahead_horizon_delta_t)
                _events = [event for event in new_chain if start_time <= event.time <= lookahead_horizon]
            else:
                _events = [event for event in new_chain if start_time <= event.time]
                
            # if len(_events) <= 0:
            #     _events = new_chain[0]
                
            event_chains.append(_events)
            
        return event_chains