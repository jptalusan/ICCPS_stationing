#!/bin/bash

session="test"

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
tmux rename-window -t $session:$window 'MCTS_ROLLOUT_900_ITER'
tmux send-keys 'conda activate py39' 'C-m'
cfile="configs/MCTS_ROLLOUT_900_ITER200"
tmux send-keys 'python run_mcts_no_inject.py -c '$cfile'' C-m

tmux split-window -h

tmux send-keys 'conda activate py39' 'C-m'
cfile="configs/MCTS_ROLLOUT_900_ITER400"
tmux send-keys 'python run_mcts_no_inject.py -c '$cfile'' C-m

tmux split-window -h

tmux send-keys 'conda activate py39' 'C-m'
cfile="configs/MCTS_ROLLOUT_900_ITER800"
tmux send-keys 'python run_mcts_no_inject.py -c '$cfile'' C-m

tmux select-pane -T 'MCTS_ROLLOUT_900_ITER'
tmux select-layout tiled

################################### WINDOW 1 ###################################

window=1
tmux new-window -t $session:$window -n 'MCTS_ROLLOUT_3600_ITER'
tmux send-keys 'conda activate py39' 'C-m'
cfile="configs/MCTS_ROLLOUT_3600_ITER200"
tmux send-keys 'python run_mcts_no_inject.py -c '$cfile'' C-m

tmux split-window -h

tmux send-keys 'conda activate py39' 'C-m'
cfile="configs/MCTS_ROLLOUT_3600_ITER400"
tmux send-keys 'python run_mcts_no_inject.py -c '$cfile'' C-m

tmux split-window -h

tmux send-keys 'conda activate py39' 'C-m'
cfile="configs/MCTS_ROLLOUT_3600_ITER800"
tmux send-keys 'python run_mcts_no_inject.py -c '$cfile'' C-m

tmux select-pane -T 'MCTS_ROLLOUT_3600_ITER'
tmux select-layout tiled

################################### WINDOW 2 ###################################

window=1
tmux new-window -t $session:$window -n '1800_7200'
tmux send-keys 'conda activate py39' 'C-m'
cfile="configs/MCTS_ROLLOUT_1800"
tmux send-keys 'python run_mcts_no_inject.py -c '$cfile'' C-m

tmux split-window -h

tmux send-keys 'conda activate py39' 'C-m'
cfile="configs/MCTS_ROLLOUT_7200"
tmux send-keys 'python run_mcts_no_inject.py -c '$cfile'' C-m

tmux select-pane -T 'MCTS_ROLLOUT_3600_ITER'
tmux select-layout tiled

tmux a -t $session

# jq '.reallocation = "true"' ./configs/BASELINE_LIMITED.json > tmp.$$.json && mv tmp.$$.json ./configs/BASELINE_LIMITED.json