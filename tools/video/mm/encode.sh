#!/usr/bin/env bash
# 多模态视频成片（从 ../encode.sh 改来）：逐帧 JPEG + 配乐 → H.264（libx264 crf 18、preset slow、yuv420p、+faststart）+ AAC 192k。
#   tools/video/mm/encode.sh [帧目录] [配乐 wav] [输出 mp4] [帧率]
# THREADS=8 限制编码线程（和别的任务共用机器时别把负载顶满）
set -euo pipefail
TITLE=${TITLE:-"AI 是怎么看图的 · 一张图怎么变成模型读得懂的“词”（Qwen3-VL-2B 真实运行）"}
OUT_DIR=/mnt/d/cjc/videos/multimodal
FRAMES=${1:-$OUT_DIR/frames}
SCORE=${2:-$OUT_DIR/score.wav}
MP4=${3:-$OUT_DIR/multimodal-v1.mp4}
FPS=${4:-30}

ffmpeg -hide_banner -loglevel warning -stats -y \
  -framerate "$FPS" -start_number 0 -i "$FRAMES/f%05d.jpg" \
  -i "$SCORE" \
  -map 0:v -map 1:a \
  -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -profile:v high -level 4.2 ${THREADS:+-threads $THREADS} \
  -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
  -c:a aac -b:a 192k -ar 48000 \
  -shortest -movflags +faststart \
  -metadata title="$TITLE" \
  "$MP4"
ffprobe -v error -show_entries format=duration,size:stream=codec_name,width,height,r_frame_rate -of compact "$MP4"
