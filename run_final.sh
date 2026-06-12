#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH -o logs/final_%j.txt
#SBATCH -e logs/final_%j.txt

# Remaining experiments after job 107494 was cancelled mid-B5.
# A9, B2, B3, B4 are DONE. This script only runs B5, B6, B7.
set -euo pipefail

source ~/miniconda3/bin/activate
conda activate difftrack
cd ~/CV_Final_Project/DiffTrack

mkdir -p logs

TAPVID=../data/tapvid_davis/tapvid_davis.pkl
EVAL_COMMON="--model cogvideox_t2v_2b --eval_dataset davis_first --tapvid_root $TAPVID \
  --resize_h 480 --resize_w 720 --chunk_frame_interval --average_overlapped_corr \
  --pipe_device cuda:0 --vis_video --tracks_leave_trace 15 --video_max_len -1 --end 4"
ATTN_COMMON="--model cogvideox_t2v_2b --eval_dataset davis_first --tapvid_root $TAPVID \
  --resize_h 480 --resize_w 720 --conf_attn_score --video_max_len 49 --device cuda:0 --end -1"

mkdir -p results/attn_maps results/limitations

echo "=== [B5] Attn map: layer=29 ts=49 (re-run; 001 was incomplete) ==="
rm -rf results/attn_maps/layer29_ts49
python analyze_real.py $ATTN_COMMON \
  --vis_layers 29 --vis_timesteps 49 \
  --output_dir ./results/attn_maps/layer29_ts49 2>&1 | tee results/attn_layer29_ts49.log
echo "[B5] DONE"

echo "=== [B6] Layer=8 (positional bias) ==="
time python evaluate_tapvid.py $EVAL_COMMON \
  --matching_layer 8 --matching_timestep 49 --inverse_step 49 \
  --output_dir ./results/limitations 2>&1 | tee results/layer8_run.log
echo "[B6] DONE"

echo "=== [B7] Attn map: layer=8 ts=49 ==="
python analyze_real.py $ATTN_COMMON \
  --vis_layers 8 --vis_timesteps 49 \
  --output_dir ./results/attn_maps/layer8_ts49 2>&1 | tee results/attn_layer8_ts49.log
echo "[B7] DONE"

echo "============================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "============================================"
echo ""
echo "Full result summary:"
grep "Mean delta_avg" \
  results/param_study/*/log.txt \
  results/param_study_no_chunk/*/log.txt \
  results/limitations/*/log.txt 2>/dev/null || true
