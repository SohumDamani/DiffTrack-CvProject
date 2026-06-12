#!/bin/bash
# New experiments: combined best config + complete timestep curve
# Run from: /home/sdama009/CV_Final_Project/DiffTrack
# GPU setup: srun -p gpu --gres=gpu:1 --mem=32g --time=6:00:00 --pty bash -l
#            conda activate difftrack

cd /home/sdama009/CV_Final_Project/DiffTrack

TAPVID=../data/tapvid_davis/tapvid_davis.pkl
COMMON="--model cogvideox_t2v_2b --eval_dataset davis_first --tapvid_root $TAPVID \
  --resize_h 480 --resize_w 720 --chunk_frame_interval --average_overlapped_corr \
  --pipe_device cuda:0 --vis_video --tracks_leave_trace 15 \
  --end 4 --video_max_len -1 \
  --output_dir ./results/new_experiments"

echo "============================================"
echo "Experiment A: Combined best (layer=5, t=10)"
echo "============================================"
python evaluate_tapvid.py $COMMON \
  --matching_layer 5 --matching_timestep 10 --inverse_step 10 \
  2>&1 | tee ./results/layer5_ts10_run.log
echo "Experiment A done."

echo ""
echo "============================================"
echo "Experiment B1: Timestep t=5 (layer=17)"
echo "============================================"
python evaluate_tapvid.py $COMMON \
  --matching_layer 17 --matching_timestep 5 --inverse_step 5 \
  2>&1 | tee ./results/layer17_ts5_run.log
echo "Experiment B1 done."

echo ""
echo "============================================"
echo "Experiment B2: Timestep t=20 (layer=17)"
echo "============================================"
python evaluate_tapvid.py $COMMON \
  --matching_layer 17 --matching_timestep 20 --inverse_step 20 \
  2>&1 | tee ./results/layer17_ts20_run.log
echo "Experiment B2 done."

echo ""
echo "============================================"
echo "Experiment C (BONUS): Combined best (layer=5, t=5)"
echo "============================================"
python evaluate_tapvid.py $COMMON \
  --matching_layer 5 --matching_timestep 5 --inverse_step 5 \
  2>&1 | tee ./results/layer5_ts5_run.log
echo "Experiment C done."

echo ""
echo "ALL EXPERIMENTS COMPLETE."
echo "Results in: ./results/new_experiments/"
echo "Logs in:    ./results/layer5_ts10_run.log"
echo "            ./results/layer17_ts5_run.log"
echo "            ./results/layer17_ts20_run.log"
echo "            ./results/layer5_ts5_run.log"
