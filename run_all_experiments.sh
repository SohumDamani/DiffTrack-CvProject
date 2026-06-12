#!/bin/bash
# Run all parameter comparison and limitation experiments sequentially.
# Each run saves to results/param_study or results/limitations.
# Usage: conda run -n difftrack bash run_all_experiments.sh

set -e
TAPVID=../data/tapvid_davis/tapvid_davis.pkl
COMMON="--model cogvideox_t2v_2b --eval_dataset davis_first --tapvid_root $TAPVID \
        --resize_h 480 --resize_w 720 --chunk_frame_interval --average_overlapped_corr \
        --pipe_device cuda:0 --vis_video --tracks_leave_trace 15 --max_videos 2"

echo "============================================"
echo "  PART A — PARAMETER COMPARISON EXPERIMENTS"
echo "============================================"

# ---- A1: Layer ablation ----
echo "[A1] Layer=5 (shallow)"
time python evaluate_tapvid.py $COMMON \
  --matching_layer 5 --matching_timestep 49 --inverse_step 49 \
  --output_dir ./results/param_study 2>&1 | tee results/layer5_run.log

echo "[A1] Layer=27 (deep)"
time python evaluate_tapvid.py $COMMON \
  --matching_layer 27 --matching_timestep 49 --inverse_step 49 \
  --output_dir ./results/param_study 2>&1 | tee results/layer27_run.log

# ---- A2: Timestep ablation ----
echo "[A2] Timestep=10 (noisy)"
time python evaluate_tapvid.py $COMMON \
  --matching_layer 17 --matching_timestep 10 --inverse_step 10 \
  --output_dir ./results/param_study 2>&1 | tee results/timestep10_run.log

echo "[A2] Timestep=30 (mid)"
time python evaluate_tapvid.py $COMMON \
  --matching_layer 17 --matching_timestep 30 --inverse_step 30 \
  --output_dir ./results/param_study 2>&1 | tee results/timestep30_run.log

# ---- A3: Chunk flags ----
echo "[A3] No --chunk_frame_interval"
time python evaluate_tapvid.py \
  --model cogvideox_t2v_2b --eval_dataset davis_first --tapvid_root $TAPVID \
  --resize_h 480 --resize_w 720 --average_overlapped_corr \
  --pipe_device cuda:0 --vis_video --tracks_leave_trace 15 --max_videos 2 \
  --matching_layer 17 --matching_timestep 49 --inverse_step 49 \
  --output_dir ./results/param_study 2>&1 | tee results/no_chunk_interval_run.log

echo "[A3] No --average_overlapped_corr"
time python evaluate_tapvid.py \
  --model cogvideox_t2v_2b --eval_dataset davis_first --tapvid_root $TAPVID \
  --resize_h 480 --resize_w 720 --chunk_frame_interval \
  --pipe_device cuda:0 --vis_video --tracks_leave_trace 15 --max_videos 2 \
  --matching_layer 17 --matching_timestep 49 --inverse_step 49 \
  --output_dir ./results/param_study 2>&1 | tee results/no_avg_overlap_run.log

echo "============================================"
echo "  PART B — LIMITATION DEMONSTRATIONS"
echo "============================================"

# ---- B2: Pure noise timestep ----
echo "[B2] Timestep=1 (pure noise)"
time python evaluate_tapvid.py $COMMON \
  --matching_layer 17 --matching_timestep 1 --inverse_step 1 \
  --output_dir ./results/limitations 2>&1 | tee results/timestep1_run.log

# ---- B3: Wrong layer (final) ----
echo "[B3] Layer=29 (final layer)"
time python evaluate_tapvid.py $COMMON \
  --matching_layer 29 --matching_timestep 49 --inverse_step 49 \
  --output_dir ./results/limitations 2>&1 | tee results/layer29_run.log

# ---- B4: Positional bias ----
echo "[B4] Layer=8 (positional bias)"
time python evaluate_tapvid.py $COMMON \
  --matching_layer 8 --matching_timestep 49 --inverse_step 49 \
  --output_dir ./results/limitations 2>&1 | tee results/layer8_run.log

echo "============================================"
echo "  PART A — ATTENTION MAP VISUALIZATIONS"
echo "============================================"

# Attention maps for key experiments (analyze_real.py limits to 2 videos internally)
for LAYER in 5 8 17 27 29; do
  TS=49
  echo "[ATTN] Layer=$LAYER Timestep=$TS"
  python analyze_real.py \
    --model cogvideox_t2v_2b \
    --vis_layers $LAYER --vis_timesteps $TS \
    --conf_attn_score \
    --eval_dataset davis_first --tapvid_root $TAPVID \
    --resize_h 480 --resize_w 720 --video_max_len 49 \
    --output_dir ./results/attn_maps/layer${LAYER}_timestep${TS} \
    --device cuda:0 2>&1 | tee results/attn_layer${LAYER}_ts${TS}.log
done

# Attention map for timestep=1 (noise baseline)
echo "[ATTN] Layer=17 Timestep=1"
python analyze_real.py \
  --model cogvideox_t2v_2b \
  --vis_layers 17 --vis_timesteps 1 \
  --conf_attn_score \
  --eval_dataset davis_first --tapvid_root $TAPVID \
  --resize_h 480 --resize_w 720 --video_max_len 49 \
  --output_dir ./results/attn_maps/layer17_timestep1 \
  --device cuda:0 2>&1 | tee results/attn_layer17_ts1.log

echo "============================================"
echo "  PART C — CAG MOTION GUIDANCE DEMO"
echo "============================================"

# CAG with pag_scale=0 (no guidance — baseline generation)
echo "[CAG] pag_scale=0 (no guidance)"
head -1 ./dataset/cag_prompts.txt > /tmp/cag_one_prompt.txt
python motion_guidance.py \
  --model_version 2b \
  --output_dir ./results/cag_demo/pag_scale0 \
  --txt_path /tmp/cag_one_prompt.txt \
  --pag_layers 13 17 21 \
  --pag_scale 0 \
  --cfg_scale 6 \
  --device cuda:0 2>&1 | tee results/cag_scale0_run.log

# CAG with pag_scale=1 (full guidance)
echo "[CAG] pag_scale=1 (full guidance)"
python motion_guidance.py \
  --model_version 2b \
  --output_dir ./results/cag_demo/pag_scale1 \
  --txt_path /tmp/cag_one_prompt.txt \
  --pag_layers 13 17 21 \
  --pag_scale 1 \
  --cfg_scale 6 \
  --device cuda:0 2>&1 | tee results/cag_scale1_run.log

echo "============================================"
echo "  ALL EXPERIMENTS DONE"
echo "============================================"
