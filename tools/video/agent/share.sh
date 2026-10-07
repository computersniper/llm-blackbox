#!/usr/bin/env bash
# 智能体视频的分享版：1080p30，两遍编码按片长算码率，目标 ≤120 MB（185 秒 × (4.6 Mbps 视频 + 192 kbps 音频) ≈ 111 MB）。
# 由 ../share.sh 改写：这支片子的帧序列本身就是 30 fps，不用再抽帧。
#   tools/video/agent/share.sh [帧目录] [配乐 wav] [输出 mp4] [视频码率]
set -euo pipefail
TITLE=${TITLE:-"只会写字的 AI，怎么自己动手干活 · 走进编程智能体的“黑箱”（Qwen3-4B 在真实沙箱里的录制）"}
OUT_DIR=/mnt/d/cjc/videos/agent
FRAMES=${1:-$OUT_DIR/frames30}
SCORE=${2:-$OUT_DIR/score.wav}
MP4=${3:-$OUT_DIR/agent-v1-share.mp4}
VB=${4:-4600k}
LOG=$OUT_DIR/x264-share

IN=(-framerate 30 -start_number 0 -i "$FRAMES/f%05d.jpg")
VOPT=(-c:v libx264 -preset slow -b:v "$VB" -maxrate 7000k -bufsize 9000k -pix_fmt yuv420p -profile:v high -level 4.1 -threads "${THREADS:-8}"
      -color_primaries bt709 -color_trc bt709 -colorspace bt709 -passlogfile "$LOG")

nice -n 10 ffmpeg -hide_banner -loglevel warning -stats -y "${IN[@]}" "${VOPT[@]}" -pass 1 -an -f null /dev/null
nice -n 10 ffmpeg -hide_banner -loglevel warning -stats -y "${IN[@]}" -i "$SCORE" -map 0:v -map 1:a "${VOPT[@]}" -pass 2 \
  -c:a aac -b:a 192k -ar 48000 -shortest -movflags +faststart \
  -metadata title="$TITLE" \
  "$MP4"
rm -f "$LOG"-0.log "$LOG"-0.log.mbtree
ffprobe -v error -show_entries format=duration,size:stream=codec_name,width,height,r_frame_rate -of compact "$MP4"
