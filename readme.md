TODO

define actions space
define rewards
MDP

s = start state
events:
    e, a = dm(s)
    s', r = env.take_action
        -> e = process_action
        -> update to next epoch
    s = s'


Action space:
option 1: only allocate and dispatch overload buses
option 2: allocate and dispatch all buses (which doesn't make too much sense since the buses would be serving trips.)