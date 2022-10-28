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