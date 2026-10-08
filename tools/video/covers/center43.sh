#!/bin/sh
# 4:3 封面的“标题居中”版（文件名加 -4x3c）：
# 推理、训练、智能体在 4:3 版基础上整体下移，让标题落到画面正中，空出的顶部用原图底色渐变补齐；
# 多模态原图标题在右边挪不到中间，改成用片中的苹果注意力热力图放大压暗做背景，标题按原封面的字体排版居中。
# 用法：先跑 poster43.sh，再 sh tools/video/covers/center43.sh
set -eu
V=/mnt/d/cjc/videos
F=$V/llm-inference/fonts
shift_c() {  # shift_c <输入> <输出> <下移像素> <底色>
  s=$3
  ffmpeg -loglevel error -y -i "$1" -f lavfi -i "color=c=$4:s=1440x$((s + 160))" -filter_complex \
    "[0:v]pad=1440:$((1080 + s)):0:$s:color=$4,crop=1440:1080:0:0[img];\
     [1:v]format=rgba,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='255*min(1,max(0,($s+160-Y)/160))'[fade];[img][fade]overlay=0:0" \
    -frames:v 1 "$2"
  echo "cover: $2"
}
shift_c "$V/llm-inference/poster-v7-4x3.png"    "$V/llm-inference/poster-v7-4x3c.png"    120 0x050506
shift_c "$V/llm-inference/poster-v7-en-4x3.png" "$V/llm-inference/poster-v7-en-4x3c.png" 120 0x050506
shift_c "$V/train-glass/poster-v1-4x3.png"      "$V/train-glass/poster-v1-4x3c.png"      88  0x040506
shift_c "$V/agent/poster-v1-4x3.png"            "$V/agent/poster-v1-4x3c.png"            50  0x060c15

# 多模态：背景是原封面里热力图卡片的图片区（苹果 + 注意力光），放大、轻微虚化、压暗、四角压暗；标题居中
P="$V/multimodal/poster-v1.png"
ffmpeg -loglevel error -y -i "$P" -filter_complex \
  "[0:v]crop=660:466:90:264,scale=1530:-2,crop=1440:1080,gblur=sigma=2.2,eq=brightness=-0.04:saturation=1.12,vignette=PI/4.5,\
   drawbox=x=0:y=0:w=iw:h=ih:color=0x050b16@0.22:t=fill,\
   drawtext=fontfile=$F/NotoSans-VF.ttf:text='I N S I D E   A   V I S I O N - L A N G U A G E   M O D E L':fontsize=22:fontcolor=0x5ef0d4:x=(w-tw)/2:y=390,\
   drawtext=fontfile=$F/NotoSerifSC-Black.otf:text='AI 是怎么看图的':fontsize=104:fontcolor=white:shadowcolor=black@0.6:shadowx=0:shadowy=4:x=(w-tw)/2:y=440,\
   drawtext=fontfile=$F/NotoSerifSC-SemiBold.otf:text='一 张 图 ， 怎 么 变 成 模 型 读 得 懂 的 “ 词 ”':fontsize=34:fontcolor=0xd6e0ee:x=(w-tw)/2:y=600" \
  -frames:v 1 "$V/multimodal/poster-v1-4x3c.png"
echo "cover: $V/multimodal/poster-v1-4x3c.png"
