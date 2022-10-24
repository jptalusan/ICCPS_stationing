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