#!/bin/bash

trap 'kill $train_pid $board_pid; exit' SIGINT SIGTERM

result="./Result-$(date +%y%m%d-%H%M%S)"
python3 main.py -d "$result" &
train_pid=$!
tensorboard --logdir=$result --reload_interval=15 &
board_pid=$!
echo $train_pid $board_pid
wait $train_pid
kill $board_pid $train_pid
