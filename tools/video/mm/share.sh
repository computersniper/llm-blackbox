#!/usr/bin/env bash
# 多模态视频的分享版（从 ../share.sh 改来）：1080p30，两遍编码控制码率，目标 ≤120 MB
# （177.5 秒 × (5.0 Mbps 视频 + 192 kbps 音频) ≈ 115 MB）。
#   tools/video/mm/share.sh [帧目录] [配乐 wav] [输出 mp4] [视频码率]
# THREADS=8 限制编码线程
set -euo pipefail
TITLE=${TITLE:-"AI 是怎么看图的 · 一张图怎么变成模型读得懂的“词”（Qwen3-VL-2B 真实运行）"}
OUT_DIR=/mnt/d/cjc/videos/multimodal
FRAMES=${1:-$OUT_DIR/frames}
SCORE=${2:-$OUT_DIR/score.wav}
MP4=${3:-$OUT_DIR/multimodal-v1-share.mp4}
VB=${4:-5000k}
LOG=$OUT_DIR/x264-share

IN=(-framerate 30 -start_number 0 -i "$FRAMES/f%05d.jpg")
VOPT=(-c:v libx264 -preset slow -b:v "$VB" -maxrate 7000k -bufsize 9000k -pix_fmt yuv420p -profile:v high -level 4.1 ${THREADS:+-threads $THREADS}
      -color_primaries bt709 -color_trc bt709 -colorspace bt709 -passlogfile "$LOG")

ffmpeg -hide_banner -loglevel warning -stats -y "${IN[@]}" "${VOPT[@]}" -pass 1 -an -f null /dev/null
ffmpeg -hide_banner -loglevel warning -stats -y "${IN[@]}" -i "$SCORE" -map 0:v -map 1:a "${VOPT[@]}" -pass 2 \
  -c:a aac -b:a 192k -ar 48000 -shortest -movflags +faststart \
  -metadata title="$TITLE" \
  "$MP4"
rm -f "$LOG"-0.log "$LOG"-0.log.mbtree
ffprobe -v error -show_entries format=duration,size:stream=codec_name,width,height,r_frame_rate -of compact "$MP4"
