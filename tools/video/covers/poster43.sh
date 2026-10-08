#!/bin/sh
# 把视频原来的横版封面（poster-*.png，1920×1080）收窄成 4:3（1440×1080），画面不重新设计：
# 推理、训练、智能体直接切掉左右空白；多模态原图左右隔得太开，把右边的标题块缩到 88% 挪近，接缝羽化。
# 用法：sh tools/video/covers/poster43.sh   （输出到原文件旁边，文件名加 -4x3）
set -eu
V=/mnt/d/cjc/videos
cut43() { ffmpeg -loglevel error -y -i "$1" -vf "crop=1440:1080:$2:0" "$3"; echo "cover: $3"; }
cut43 "$V/llm-inference/poster-v7.png"    330 "$V/llm-inference/poster-v7-4x3.png"
cut43 "$V/llm-inference/poster-v7-en.png" 330 "$V/llm-inference/poster-v7-en-4x3.png"
cut43 "$V/train-glass/poster-v1.png"      240 "$V/train-glass/poster-v1-4x3.png"
cut43 "$V/agent/poster-v1.png"            245 "$V/agent/poster-v1-4x3.png"

# 多模态：底图取左边 1440（含热力图卡片）；先用边缘羽化的深色块盖掉原标题，再把标题块（原图 x 930–1720）缩到 88% 放到右侧
P="$V/multimodal/poster-v1.png"
ffmpeg -loglevel error -y -i "$P" -f lavfi -i "color=c=0x060d18:s=720x500" -filter_complex \
  "[0:v]crop=1440:1080:20:0[base];\
   [1:v]format=rgba,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='255*min(1,X/90)*min(1,Y/70)*min(1,(500-Y)/70)'[mask];\
   [0:v]crop=790:420:930:320,scale=695:-2,format=rgba,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='255*min(1,X/120)*min(1,(695-X)/50)*min(1,Y/50)*min(1,(H-Y)/50)'[title];\
   [base][mask]overlay=720:285[b2];[b2][title]overlay=722:345" \
  -frames:v 1 "$V/multimodal/poster-v1-4x3.png"
echo "cover: $V/multimodal/poster-v1-4x3.png"
