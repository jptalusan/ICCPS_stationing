# Stationing Simulator for MCTS

## Contents
0. [Setting up workspace](#workspace)
1. [Generate models](#generate-models)
2. [Usage](#usage)

## Workspace
1. Clone the repo:
    ``` bash
    git clone https://github.com/jptalusan/mta_simulator_redo.git
    git checkout dispatch_heuristic
    ```
2. Download the required files (will just put a google drive link probably):
    * apc_weather_gtfs_20220921.parquet: combined dataset of APC, weather and GTFS ([see here to generate](https://github.com/jptalusan/wego_occupancy_clean/blob/main/notebooks/preprocessing.ipynb))
    * occupancy prediction models: generated by running same_day.ipynb in the link above. 
    * School Breaks (2019-2022).pkl
    * US Holiday Dates (2004-2021).csv

## Generate models
1. Run `data_generation/run_all.py` to generate all required common files.
    * `pair_dd_tt`
    * `travel_time_by_scheduled_time`
    * `sampled_travel_times_dict`
    * `davidson_graph.graphml`
    * `pair_tt_dd_stops`
    * `stops_tt_dd_node_dict`
    * `stops_tt_dd_dict`
    * `stops_node_matching_dict`
2. Run `data_generation/generate_day_trips.py` to generate trip and vehicle plans as well as chains.
3. Run `data_generation/step0_generator` to generate:
    * `disruption_probabilities`
    * Config and Execute bash scripts for testing.
4. Run `stops_clustering` to generate possible stationing locations.

## Usage
1. Create a copy of the `TEST` folder in `code_root/experiments` and rename.
2. Place generate config files in `FOLDER/configs`.
3. From `code_root/experiments/FOLDER`, run `python run_mcts_no_inject.py -c configs/CONFIG_NAME`
4. Once done, it will generate logs inside `logs`.