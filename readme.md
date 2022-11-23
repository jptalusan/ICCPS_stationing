# Stationing Simulator for MCTS

```
s = start state
events:
    e, a = dm(s)
    s', r = env.take_action
        -> e = process_action
        -> update to next epoch  
    s = s'  
```

Action space:  
option 1: only allocate and dispatch overload buses  
option 2: allocate and dispatch all buses (which doesn't make too much sense since the buses would be serving trips.)  


Important TODO 10-21-2022  
I might need to remove the assert on event times in EnvModelFast:30 it fails at all vehicles event 323.  
I removed raise in EmpiricalLookup  
    - i need to regenerate the tt_dd_stops files, for some reason, it did not include ALL pairs. weird.  
    - 10/24: I found out the issue, i was using itertools combination instead of product. Have not run it yet, but will soon.  


ISSUE:
leave stops - only when busses arrrive
no obs until then

state - add some helper variable

limiting decision epochs

data driven way might be too complicte
rule based approach first 
    - ie. if decision epoch in last 5 minutes/60min, then don't do, unless something critical happens
    - get some data

double check on passenger leaving by stops (without cleanup at end step)


only decision epoch when BUS arrives!!

each node in the tree only should be when bus arrives, not passenger
who arrives, who left at the stops

just because an event happened, it doesnt mean its a decision epoch

speed up by 50% at least
since we also decrease tree search by reducing decision nodes

try outright removing them from the event list and just create a datastructure that looks up the people at that stop when a bus arrives. then update the state of that stop.

Could not find tt/dd for 202314951/EDG9AWN to 202256411/EDGPEDWF.
Could not find tt/dd for 202256411/EDGPEDWF to 202314983/EDG12AWN.
Could not find tt/dd for 202314951/EDG9AWN to 202256411/EDGPEDWF.
Could not find tt/dd for 202256411/EDGPEDWF to 202314983/EDG12AWN.

issue: buses arriving very late and still picking up passengers

DONE:
Already added time points, do i also have to do it in the rollouts?

FIXED:
Getting stuck with dispatching when low capacity
    - Fixd by setting IDLE before starttrip in action but...
    - New bug: next trip not working now.

Rollout:
SendNearestDispatch if a vehicle is broken, else do nothing.
However, breakdown events are not generated in the event chains.

TODO:
Left as is for now since reallocation happens to frequently already.
Need to fix the allocation again!!!
Need to give bus under reallocation the ability to skip what they are doing and head for other allocation or broken stops
    - questions:
    how will we know where the bus currently is? interpolation?
    how will we know compute?


11-06-2022

Stationing
0. Brought back the config file. (DONE)
1. Set interval for decision epochs (GLOBAL not per trip) (DONE)
2. Remove the overflow buses first (DONE)
    * Try to change the code to remove "VEHICLE_START_TRIP" (DONE)
    * Only breakdown, and bus arrival events must be covered in EVENTS (DONE)
3. Bring back the overflow (DONE)
    * Remove ALLOCATION first and use T_STATE_CHANGE (DONE)
    * Set them as IN_TRANSIT and ARRIVAL_AT_STOP when action is taken (DISPATCH AND BROKEN)
    * loop through ALL OVERFLOW buses in each bus arrival events (DONE) only for allocation buses
4. Bring back ALLOCATION (DONE)
    * ALLOCATION ACTION maintains the bus to be IDLE and can be used for dispatch or broken (DONE)

Issues:
10 vehicle: issue with ASSERT (have solution but untested)
Max(scheduled, t_state_change)
EnvironmentalModelFast:258 assert passenger_arrival_time <= bus_arrival_time

Some uncertainty on the overall passenger count
* will have to count the actual in the sampled distribution for ALL buses.
* A bit overhauled the environemtnmodelfast. Passenger pick up. I think it makes sense.

Places where actions are generated/reprocessed. Confusion on list of tuples of dicts or list of dicts.
1. DecisionEnvironmentDynamics::generate_possible_actions
2. SendNearestDispatchPolicy::select_overload_to_dispatch
3. ModularMCTS::pick_expand_action
4. Simulator::run_simulation
5. DecisionMaker::get_action

Do overflow buses also causes decision epochs?
The reason why the rollout time grows as time passes is that more buses are active.
Problem: Buses keep getting assigned to the same trip even when others are assigned to it already.
    - can handle with rollouts, but will increaese time.
problem: How to decrease rollout time? how to approximate the state updates?
Question: Should the rollout still be generating new events? (i think yes.)
    - But the issue is even though i set a horizon limit, i still generate all the new events until the end?! (of at least the buses in the truncated events)
    - i should limit this new events to within the horizon too!


# TODO:
0. Set up baseline (greedy, send nearest)
1. Set up custom interval events (which control when simulator will use decision maker)
2. Setup combinatorial with intervals
    * Decide which trip/trips to send overload buses to
    * Next action, decide which bus to send to which trip/trips (start with nearest (iterative))


11-22-2022
* Changed ROLLOUT to behave similar to GreedyCoordinator
* Added configs to control code easier
* Added random.seeds for testability
* Fixed issue with `update_event = self.event_queues.pop(0)` that was causing events to be missed.
* Fixed conditions which failed to handle `LEAVE_PASSENGER_BEHIND` events.
* Created experiments folder for easier testing.