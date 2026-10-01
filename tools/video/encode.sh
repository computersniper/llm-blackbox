#!/usr/bin/env bash
# 把逐帧渲染的 JPEG 和配乐合成成片：H.264（libx264 crf 18、preset slow、yuv420p、+faststart）+ AAC 192k。
#   tools/video/encode.sh [帧目录] [配乐 wav] [输出 mp4] [帧率]
set -euo pipefail
OUT_DIR=/mnt/d/cjc/videos/llm-inference
FRAMES=${1:-$OUT_DIR/frames60}
SCORE=${2:-$OUT_DIR/score.wav}
MP4=${3:-$OUT_DIR/qwen3-inference.mp4}
FPS=${4:-60}

ffmpeg -hide_banner -loglevel warning -stats -y \
  -framerate "$FPS" -start_number 0 -i "$FRAMES/f%05d.jpg" \
  -i "$SCORE" \
  -map 0:v -map 1:a \
  -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -profile:v high -level 4.2 \
  -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
  -c:a aac -b:a 192k -ar 48000 \
  -shortest -movflags +faststart \
  -metadata title="AI 的一个字是怎么思考出来的 · 走进大模型推理的“黑箱”（Qwen3-0.6B 真实推理）" \
  "$MP4"
ffprobe -v error -show_entries format=duration,size:stream=codec_name,width,height,r_frame_rate -of compact "$MP4"
