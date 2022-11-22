#!/bin/bash

# tmux new-session -d -s pwd
# tmux send-keys pwd C-m
# tmux a -t pwd

session="tests"

tmux kill-session -t $session
tmux new-session -d -s $session

window=0
tmux rename-window -t $session:$window '1'
tmux send-keys 'conda activate py39' 'C-m' 'cd /media/seconddrive/JP/gits/mta_simulator_redo/code_root/20220305' 'C-m'
tmux send-keys 'python run_mcts.py -c tests/config_1B_800_7200RO' C-m
tmux select-pane -T 'config_1B_800_7200RO'

tmux split-window -h
tmux send-keys 'conda activate py39' 'C-m' 'cd /media/seconddrive/JP/gits/mta_simulator_redo/code_root/20220305' 'C-m'
tmux send-keys 'python run_mcts.py -c configs/1B_200_LIMITED' C-m

tmux split-window -h
tmux send-keys 'conda activate py39' 'C-m' 'cd /media/seconddrive/JP/gits/mta_simulator_redo/code_root/20220305' 'C-m'
tmux send-keys 'python run_mcts_no_inject.py -c configs/config_1B_800_5400RO' C-m
# tmux select-pane -T 'config_1B_800_5400RO'

tmux split-window -h
tmux send-keys 'conda activate py39' 'C-m' 'cd /media/seconddrive/JP/gits/mta_simulator_redo/code_root/20220305' 'C-m'
tmux send-keys 'python run_mcts_no_inject.py -c configs/BASELINE_FULL' C-m
tmux send-keys 'python run_mcts_no_inject.py -c configs/BASELINE_LIMITED_NO_OVER' C-m
tmux send-keys 'python run_mcts_no_inject.py -c configs/BASELINE_LIMITED' C-m

tmux send-keys 'python run_mcts.py -c configs/BASELINE_FULL' C-m
tmux send-keys 'python run_mcts.py -c configs/BASELINE_LIMITED_NO_OVER' C-m
tmux send-keys 'python run_mcts.py -c configs/BASELINE_LIMITED' C-m
tmux select-pane -T 'BASELINE'

# tmux split-window -h
# tmux send-keys 'conda activate py39' 'C-m' 'cd /media/seconddrive/JP/gits/mta_simulator_redo/code_root' 'C-m'
# tmux send-keys 'python run_mcts.py -c tests/config_1A_800' C-m
# tmux select-pane -T '1A_800'

# tmux split-window -h
# tmux send-keys 'conda activate py39' 'C-m' 'cd /media/seconddrive/JP/gits/mta_simulator_redo/code_root' 'C-m'
# tmux send-keys 'python run_mcts.py -c tests/config_1A_400_THREAD' C-m

tmux select-layout tiled

window=1
tmux new-window -t $session:$window -n '1B'
tmux send-keys 'conda activate py39' 'C-m' 'cd /media/seconddrive/JP/gits/mta_simulator_redo/code_root' 'C-m'
tmux send-keys 'python run_mcts_20211018.py -c tests/config_1A_400_THREAD' C-m

tmux split-window -h
tmux send-keys 'conda activate py39' 'C-m' 'cd /media/seconddrive/JP/gits/mta_simulator_redo/code_root' 'C-m'
tmux send-keys 'python run_mcts_20211018.py -c tests/config_1B_400_THREAD' C-m

tmux split-window -h
tmux send-keys 'conda activate py39' 'C-m' 'cd /media/seconddrive/JP/gits/mta_simulator_redo/code_root' 'C-m'
tmux send-keys 'python run_mcts_20211018.py -c tests/config_2A_400_THREAD' C-m

tmux split-window -h
tmux send-keys 'conda activate py39' 'C-m' 'cd /media/seconddrive/JP/gits/mta_simulator_redo/code_root' 'C-m'
tmux send-keys 'python run_mcts.py -c tests/config_1B_800_ALL' C-m

# tmux split-window -h
# tmux send-keys 'conda activate py39' 'C-m' 'cd /media/seconddrive/JP/gits/mta_simulator_redo/code_root' 'C-m'
# tmux send-keys 'python run_mcts.py -c tests/config_1B_400_THREAD' C-m

tmux select-layout tiled

# window=2
# tmux new-window -t $session:$window -n '2A'
# tmux send-keys 'conda activate py39' 'C-m' 'cd /media/seconddrive/JP/gits/mta_simulator_redo/code_root' 'C-m'
# tmux send-keys 'python run_mcts.py -c tests/config_2A_200' C-m

# tmux split-window -h
# tmux send-keys 'conda activate py39' 'C-m' 'cd /media/seconddrive/JP/gits/mta_simulator_redo/code_root' 'C-m'
# tmux send-keys 'python run_mcts.py -c tests/config_2A_400' C-m

# tmux split-window -h
# tmux send-keys 'conda activate py39' 'C-m' 'cd /media/seconddrive/JP/gits/mta_simulator_redo/code_root' 'C-m'
# tmux send-keys 'python run_mcts.py -c tests/config_2A_600' C-m

# tmux split-window -h
# tmux send-keys 'conda activate py39' 'C-m' 'cd /media/seconddrive/JP/gits/mta_simulator_redo/code_root' 'C-m'
# tmux send-keys 'python run_mcts.py -c tests/config_2A_800' C-m

# tmux split-window -h
# tmux send-keys 'conda activate py39' 'C-m' 'cd /media/seconddrive/JP/gits/mta_simulator_redo/code_root' 'C-m'
# tmux send-keys 'python run_mcts.py -c tests/config_1B_400_THREAD' C-m

# tmux select-layout tiled

tmux a -t $session