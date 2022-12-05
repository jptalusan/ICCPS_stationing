1. Generate 4 hour window (limit trips as well) see most busy window
* 8am - 12pm
* generate trip plan and vehicle plan for 5 hours

2. Split chains into:
* Train: for c, iterations
* Lookahead: sensitivity for wait times, # of overload buses
* Test: pick some values from look ahead sensitivity and use for full day on test set.
* For all of this i'll test against the baseline (for each chain)

## MODS
Line 338:DecisionMaker
```
def get_passenger_arrival_distributions(self, chain_count=1):
    chain_dir = f'{self.base_dir}/chains/{self.starting_date}_TRAIN'
```

Line 17: EmpiricalTravelModelLookup
```
config_path = f'{base_dir}/trip_plan_{date_str}_limited.json'
```