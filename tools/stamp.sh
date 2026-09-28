#!/bin/sh
# 给 CSS/JS 的引用加上 ?v=<版本>，避免浏览器缓存把新旧模块混在一起。
# 用法：sh tools/stamp.sh <站点目录> <版本号>
# 覆盖所有页面（首页和 train/、multimodal/ 等子页面）里的相对引用，以及模块之间的静态 / 动态 import。
set -eu
dir=$1
v=$2
find "$dir" -name '*.html' -exec sed -i -E "s#((href|src)=\"[^\":?]*\.(css|js))\"#\1?v=$v\"#g" {} +
find "$dir" -name '*.js' -exec sed -i -E "s#(from ['\"][^'\"?]*\.js)(['\"])#\1?v=$v\2#g; s#(import\(['\"][^'\"?]*\.js)(['\"]\))#\1?v=$v\2#g" {} +
