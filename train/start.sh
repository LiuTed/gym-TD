#!/bin/bash

trap 'kill $train_pid $board_pid; exit' SIGINT SIGTERM

d="$(date +%y%m%d-%H%M%S)"
result="./Result-$d"
ckpt="./ckpt-$d"
python3 main.py $* -d "$result" -s "$ckpt" &
train_pid=$!
tensorboard --logdir=$result --reload_interval=15 --bind_all &
board_pid=$!
echo $train_pid $board_pid
wait $train_pid
kill $board_pid $train_pid
