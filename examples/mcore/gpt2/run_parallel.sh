#!/bin/bash
set -e

mkdir -p logs

echo "Starting Batch 1 (LR 1e-3)..."
# Port 6000, Devices 0-7
bash examples/mcore/gpt2/pretrain_gpt2_small_adamuon.sh 6000 1e-3 0,1,2,3,4,5,6,7 > logs/launch_small_adamuon_1e-3.log 2>&1 &
# Port 6001, Devices 8-15
bash examples/mcore/gpt2/pretrain_gpt2_small_muon.sh 6001 1e-3 8,9,10,11,12,13,14,15 > logs/launch_small_muon_1e-3.log 2>&1 &
wait

echo "Starting Batch 2 (LR 1e-3)..."
# Port 6002, Devices 0-7
bash examples/mcore/gpt2/pretrain_gpt2_small_adam.sh 6002 1e-3 0,1,2,3,4,5,6,7 > logs/launch_small_adam_1e-3.log 2>&1 &
# Port 6003, Devices 8-15
bash examples/mcore/gpt2/pretrain_gpt2_small_sophiamuon.sh 6003 1e-3 8,9,10,11,12,13,14,15 > logs/launch_small_sophiamuon_1e-3.log 2>&1 &
wait

echo "Starting Batch 3 (LR 3e-4)..."
# Port 6004, Devices 0-7
bash examples/mcore/gpt2/pretrain_gpt2_small_adamuon.sh 6004 3e-4 0,1,2,3,4,5,6,7 > logs/launch_small_adamuon_3e-4.log 2>&1 &
# Port 6005, Devices 8-15
bash examples/mcore/gpt2/pretrain_gpt2_small_muon.sh 6005 3e-4 8,9,10,11,12,13,14,15 > logs/launch_small_muon_3e-4.log 2>&1 &
wait

echo "Starting Batch 4 (LR 3e-4)..."
# Port 6006, Devices 0-7
bash examples/mcore/gpt2/pretrain_gpt2_small_adam.sh 6006 3e-4 0,1,2,3,4,5,6,7 > logs/launch_small_adam_3e-4.log 2>&1 &
# Port 6007, Devices 8-15
bash examples/mcore/gpt2/pretrain_gpt2_small_sophiamuon.sh 6007 3e-4 8,9,10,11,12,13,14,15 > logs/launch_small_sophiamuon_3e-4.log 2>&1 &
wait

echo "All tasks completed!"
