#!/bin/sh
# 给 CSS/JS/字体的引用加上 ?v=<版本>，避免浏览器缓存把新旧文件混在一起（服务器对这些文件设了长缓存）。
# 用法：sh tools/stamp.sh <站点目录> <版本号>
# 覆盖所有页面（首页和 train/、multimodal/、agent/ 等子页面）里的相对引用、CSS 里的字体 url()，以及模块之间的静态 / 动态 import
# （包括压缩过、from 后面没有空格的写法，如 three.module.js 里的 from"./three.core.js"）。
# 每个模块的 URL 因此都是“路径?v=<版本>”：页面里预取模块（modulepreload）时按同样的规则拼 URL，才能和 import 命中同一份。
set -eu
dir=$1
v=$2
find "$dir" -name '*.html' -exec sed -i -E "s#((href|src)=\"[^\":?]*\.(css|js|woff2))\"#\1?v=$v\"#g" {} +
find "$dir" -name '*.css' -exec sed -i -E "s#(url\(\"[^\":?]*\.woff2)\"#\1?v=$v\"#g" {} +
find "$dir" -name '*.js' -exec sed -i -E "s#(from ?['\"][^'\"?]*\.js)(['\"])#\1?v=$v\2#g; s#(import\(['\"][^'\"?]*\.js)(['\"]\))#\1?v=$v\2#g" {} +
