#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH -o logs/remaining_%j.txt
#SBATCH -e logs/remaining_%j.txt

# DiffTrack remaining experiments — 4 videos each, attn maps alongside every tracking run.
# Living log: each section prints "[Xn] DONE" on success.
# To resume after a crash:
#   1. Check logs/remaining_<jobid>.txt for "[Xn] DONE" lines
#   2. Comment out those sections below and re-run
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

mkdir -p results/attn_maps results/param_study_no_chunk results/limitations

# ── A4–A8: DONE ──────────────────────────────────────────────────────────────
echo "[A4] DONE (skipped — results already in results/attn_maps/layer{5,17,27}_ts49)"
echo "[A5] DONE (skipped — results/param_study/layer[17]_timestep[10]_noiseFalse: delta_avg=13.1)"
echo "[A6] DONE (skipped — results/attn_maps/layer17_ts10 exists)"
echo "[A7] DONE (skipped — results/param_study/layer[17]_timestep[30]_noiseFalse: delta_avg=47.8)"
echo "[A8] DONE (skipped — results/attn_maps/layer17_ts30 exists)"

# ── A9: No --chunk_frame_interval ────────────────────────────────────────────
# Uses a separate output_dir so it doesn't collide with baseline (same layer/ts).
# Previous partial run had only 2 frames; cleaned up and restarting with --end 4.
echo "=== [A9] No --chunk_frame_interval ==="
rm -rf results/param_study_no_chunk/layer\[17\]_timestep\[49\]_noiseFalse
time python evaluate_tapvid.py \
  --model cogvideox_t2v_2b --eval_dataset davis_first --tapvid_root $TAPVID \
  --resize_h 480 --resize_w 720 --average_overlapped_corr \
  --pipe_device cuda:0 --vis_video --tracks_leave_trace 15 --video_max_len -1 --end 4 \
  --matching_layer 17 --matching_timestep 49 --inverse_step 49 \
  --output_dir ./results/param_study_no_chunk 2>&1 | tee results/no_chunk_run.log
echo "[A9] DONE"

# ── B2: Timestep=1 (clean/denoised end — per paper t=1 is fully denoised) ────
echo "=== [B2] Timestep=1 (clean denoised latent) ==="
time python evaluate_tapvid.py $EVAL_COMMON \
  --matching_layer 17 --matching_timestep 1 --inverse_step 1 \
  --output_dir ./results/limitations 2>&1 | tee results/timestep1_run.log
echo "[B2] DONE"

# ── B3: Attention map for timestep=1 ─────────────────────────────────────────
echo "=== [B3] Attn map: layer=17 ts=1 ==="
python analyze_real.py $ATTN_COMMON \
  --vis_layers 17 --vis_timesteps 1 \
  --output_dir ./results/attn_maps/layer17_ts1 2>&1 | tee results/attn_layer17_ts1.log
echo "[B3] DONE"

# ── B4: Layer=29 final layer breakdown ───────────────────────────────────────
echo "=== [B4] Layer=29 (final/over-abstracted layer) ==="
time python evaluate_tapvid.py $EVAL_COMMON \
  --matching_layer 29 --matching_timestep 49 --inverse_step 49 \
  --output_dir ./results/limitations 2>&1 | tee results/layer29_run.log
echo "[B4] DONE"

# ── B5: Attention map for layer=29 ───────────────────────────────────────────
echo "=== [B5] Attn map: layer=29 ts=49 ==="
python analyze_real.py $ATTN_COMMON \
  --vis_layers 29 --vis_timesteps 49 \
  --output_dir ./results/attn_maps/layer29_ts49 2>&1 | tee results/attn_layer29_ts49.log
echo "[B5] DONE"

# ── B6: Layer=8 positional bias demonstration ────────────────────────────────
echo "=== [B6] Layer=8 (positional bias) ==="
time python evaluate_tapvid.py $EVAL_COMMON \
  --matching_layer 8 --matching_timestep 49 --inverse_step 49 \
  --output_dir ./results/limitations 2>&1 | tee results/layer8_run.log
echo "[B6] DONE"

# ── B7: Attention map for layer=8 ────────────────────────────────────────────
echo "=== [B7] Attn map: layer=8 ts=49 ==="
python analyze_real.py $ATTN_COMMON \
  --vis_layers 8 --vis_timesteps 49 \
  --output_dir ./results/attn_maps/layer8_ts49 2>&1 | tee results/attn_layer8_ts49.log
echo "[B7] DONE"

echo "============================================"
echo "  ALL REMAINING EXPERIMENTS DONE"
echo "============================================"
echo ""
echo "Quick result summary:"
grep "Mean delta_avg" \
  results/param_study/*/log.txt \
  results/param_study_no_chunk/*/log.txt \
  results/limitations/*/log.txt 2>/dev/null || true
