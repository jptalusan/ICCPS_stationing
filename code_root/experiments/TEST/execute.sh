#!/bin/bash

session="c_search_3600"

# Check if the session exists, discarding output
# We can check $? for the exit status (zero for success, non-zero for failure)
tmux has-session -t $session 2>/dev/null

if [ $? != 0 ]; then
  # Set up your session
  tmux new-session -d -s $session
else
  tmux kill-session -t $session
fi

# ################################### WINDOW 0 ###################################

window=0
tmux rename-window -t $session:$window '1'
tmux send-keys 'conda activate py39' 'C-m'
cfile="configs/mean"
logName=${cfile#*/}
tmux send-keys 'python run_mcts_no_inject.py -c '$cfile'' C-m
tmux send-keys 'python emailer.py -c '$cfile'' C-m
tmux send-keys 'tmux capture-pane -pJ -S - > 'logs/$logName'.log' C-m

cfile="configs/percentile_20"
logName=${cfile#*/}
tmux send-keys 'python run_mcts_no_inject.py -c '$cfile'' C-m
tmux send-keys 'tmux capture-pane -pJ -S - > 'logs/$logName'.log' C-m

cfile="configs/percentile_50"
logName=${cfile#*/}
tmux send-keys 'python run_mcts_no_inject.py -c '$cfile'' C-m
tmux send-keys 'tmux capture-pane -pJ -S - > 'logs/$logName'.log' C-m

cfile="configs/percentile_100"
logName=${cfile#*/}
tmux send-keys 'python run_mcts_no_inject.py -c '$cfile'' C-m
tmux send-keys 'tmux capture-pane -pJ -S - > 'logs/$logName'.log' C-m

tmux split-window -h

tmux send-keys 'conda activate py39' 'C-m'
cfile="configs/percentile_10"
logName=${cfile#*/}
tmux send-keys 'python run_mcts_no_inject.py -c '$cfile'' C-m
tmux send-keys 'tmux capture-pane -pJ -S - > 'logs/$logName'.log' C-m

cfile="configs/percentile_30"
logName=${cfile#*/}
tmux send-keys 'python run_mcts_no_inject.py -c '$cfile'' C-m
tmux send-keys 'tmux capture-pane -pJ -S - > 'logs/$logName'.log' C-m

cfile="configs/percentile_60"
logName=${cfile#*/}
tmux send-keys 'python run_mcts_no_inject.py -c '$cfile'' C-m
tmux send-keys 'tmux capture-pane -pJ -S - > 'logs/$logName'.log' C-m

cfile="configs/no_c"
logName=${cfile#*/}
tmux send-keys 'python run_mcts_no_inject.py -c '$cfile'' C-m
tmux send-keys 'tmux capture-pane -pJ -S - > 'logs/$logName'.log' C-m

tmux split-window -h

tmux send-keys 'conda activate py39' 'C-m'
cfile="configs/percentile_90"
logName=${cfile#*/}
tmux send-keys 'python run_mcts_no_inject.py -c '$cfile'' C-m
tmux send-keys 'tmux capture-pane -pJ -S - > 'logs/$logName'.log' C-m

cfile="configs/percentile_40"
logName=${cfile#*/}
tmux send-keys 'python run_mcts_no_inject.py -c '$cfile'' C-m
tmux send-keys 'tmux capture-pane -pJ -S - > 'logs/$logName'.log' C-m

cfile="configs/percentile_80"
logName=${cfile#*/}
tmux send-keys 'python run_mcts_no_inject.py -c '$cfile'' C-m
tmux send-keys 'tmux capture-pane -pJ -S - > 'logs/$logName'.log' C-m

tmux select-pane -T '1'
tmux select-layout tiled

# ################################### WINDOW 1 ###################################

# window=1
# tmux new-window -t $session:$window -n 'UCT10'
# tmux send-keys 'conda activate py39' 'C-m'
# cfile="configs/BL"
# tmux send-keys 'python run_mcts_no_inject.py -c '$cfile'' C-m

# tmux split-window -h

# tmux send-keys 'conda activate py39' 'C-m'
# cfile="configs/100U_400I"
# tmux send-keys 'python run_mcts_no_inject.py -c '$cfile'' C-m

# tmux split-window -h

# tmux send-keys 'conda activate py39' 'C-m'
# cfile="configs/100U_800I"
# tmux send-keys 'python run_mcts_no_inject.py -c '$cfile'' C-m

# tmux split-window -h

# tmux send-keys 'conda activate py39' 'C-m'
# cfile="configs/100U_1000I"
# tmux send-keys 'python run_mcts_no_inject.py -c '$cfile'' C-m

# tmux select-pane -T 'UCT10'
# tmux select-layout tiled

# ################################### WINDOW 2 ###################################

# window=2
# tmux new-window -t $session:$window -n 'UCT144'
# tmux send-keys 'conda activate py39' 'C-m'
# cfile="configs/144U_200I"
# tmux send-keys 'python run_mcts_no_inject.py -c '$cfile'' C-m

# tmux split-window -h

# tmux send-keys 'conda activate py39' 'C-m'
# cfile="configs/144U_400I"
# tmux send-keys 'python run_mcts_no_inject.py -c '$cfile'' C-m

# tmux split-window -h

# tmux send-keys 'conda activate py39' 'C-m'
# cfile="configs/144U_800I"
# tmux send-keys 'python run_mcts_no_inject.py -c '$cfile'' C-m

# tmux split-window -h

# tmux send-keys 'conda activate py39' 'C-m'
# cfile="configs/144U_1000I"
# tmux send-keys 'python run_mcts_no_inject.py -c '$cfile'' C-m

# tmux select-pane -T 'UCT144'
# tmux select-layout tiled

# ################################### WINDOW 3 ###################################

# window=1
# tmux new-window -t $session:$window -n 'BL'

# tmux send-keys 'conda activate py39' 'C-m'
# cfile="configs/BL"
# logName=${cfile#*/}
# tmux send-keys 'python run_mcts_no_inject.py -c '$cfile'' C-m
# tmux send-keys 'tmux capture-pane -pJ -S - > 'logs/$logName'.log' C-m

tmux a -t $session

# jq '.reallocation = "true"' ./configs/BASELINE_LIMITED.json > tmp.$$.json && mv tmp.$$.json ./configs/BASELINE_LIMITED.json