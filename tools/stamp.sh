#!/bin/sh
# 给 CSS/JS 的引用加上 ?v=<版本>，避免浏览器缓存把新旧模块混在一起。
# 用法：sh tools/stamp.sh <站点目录> <版本号>
set -eu
dir=$1
v=$2
sed -i "s#\(href=\"css/[^\"?]*\.css\)\"#\1?v=$v\"#g; s#\(src=\"js/main\.js\)\"#\1?v=$v\"#" "$dir/index.html"
find "$dir/js" -name '*.js' -exec sed -i "s#\(from '[^'?]*\.js\)'#\1?v=$v'#g" {} +
