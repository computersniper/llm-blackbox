#!/usr/bin/env bash
# 分享用的小体积版本：1080p、30fps，两遍编码控制码率，目标 ≤150 MB（305 秒 × (3.6 Mbps 视频 + 192 kbps 音频) ≈ 145 MB）。
# 音频不降到 128k：低码率的 AAC 在密集段落会冲出 0 dBFS 以上的峰值
#   tools/video/share.sh [帧目录] [配乐 wav] [输出 mp4] [视频码率]
set -euo pipefail
OUT_DIR=/mnt/d/cjc/videos/llm-inference
FRAMES=${1:-$OUT_DIR/frames60}
SCORE=${2:-$OUT_DIR/score.wav}
MP4=${3:-$OUT_DIR/qwen3-inference-v4-share.mp4}
VB=${4:-3600k}
LOG=$OUT_DIR/x264-share

# 60fps 的帧序列按 30fps 取（隔一帧取一帧）
IN=(-framerate 60 -start_number 0 -i "$FRAMES/f%05d.jpg")
VOPT=(-vf fps=30 -c:v libx264 -preset slow -b:v "$VB" -maxrate 7000k -bufsize 9000k -pix_fmt yuv420p -profile:v high -level 4.1
      -color_primaries bt709 -color_trc bt709 -colorspace bt709 -passlogfile "$LOG")

ffmpeg -hide_banner -loglevel warning -stats -y "${IN[@]}" "${VOPT[@]}" -pass 1 -an -f null /dev/null
ffmpeg -hide_banner -loglevel warning -stats -y "${IN[@]}" -i "$SCORE" -map 0:v -map 1:a "${VOPT[@]}" -pass 2 \
  -c:a aac -b:a 192k -ar 48000 -shortest -movflags +faststart \
  -metadata title="AI 的一个字是怎么思考出来的 · 走进大模型推理的“黑箱”（Qwen3-0.6B 真实推理）" \
  "$MP4"
rm -f "$LOG"-0.log "$LOG"-0.log.mbtree
ffprobe -v error -show_entries format=duration,size:stream=codec_name,width,height,r_frame_rate -of compact "$MP4"
