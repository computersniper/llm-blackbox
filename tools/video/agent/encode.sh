#!/usr/bin/env bash
# 智能体视频成片：帧 + 配乐 → H.264（libx264 crf 18、preset slow、yuv420p、+faststart）+ AAC 192k。
# 由 ../encode.sh 改写：默认路径换成 /mnt/d/cjc/videos/agent/、30 fps，并限制编码线程（WSL 下瞬时负载太高会整机崩溃）。
#   tools/video/agent/encode.sh [帧目录] [配乐 wav] [输出 mp4] [帧率]
set -euo pipefail
TITLE=${TITLE:-"只会写字的 AI，怎么自己动手干活 · 走进编程智能体的“黑箱”（Qwen3-4B 在真实沙箱里的录制）"}
OUT_DIR=/mnt/d/cjc/videos/agent
FRAMES=${1:-$OUT_DIR/frames30}
SCORE=${2:-$OUT_DIR/score.wav}
MP4=${3:-$OUT_DIR/agent-v1.mp4}
FPS=${4:-30}
THREADS=${THREADS:-8}

nice -n 10 ffmpeg -hide_banner -loglevel warning -stats -y \
  -framerate "$FPS" -start_number 0 -i "$FRAMES/f%05d.jpg" \
  -i "$SCORE" \
  -map 0:v -map 1:a \
  -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -profile:v high -level 4.2 -threads "$THREADS" \
  -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
  -c:a aac -b:a 192k -ar 48000 \
  -shortest -movflags +faststart \
  -metadata title="$TITLE" \
  "$MP4"
ffprobe -v error -show_entries format=duration,size:stream=codec_name,width,height,r_frame_rate -of compact "$MP4"
