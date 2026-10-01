#!/bin/sh
# 在服务器上执行一次（幂等）：建 bare 仓库、装钩子、把站点目录挂到博客根目录下。
#   scp deploy/post-receive <用户>@<服务器>:/tmp/blackbox-post-receive
#   ssh <用户>@<服务器> 'sh -s' < deploy/setup-server.sh
# 之后本地 `git push deploy main` 即可发布。
set -eu
ROOT=/srv/blackbox
WEB=/srv/www.caijiechao.com/frontend/blackbox
HOOK_SRC=${1:-/tmp/blackbox-post-receive}

mkdir -p "$ROOT/releases"
[ -d "$ROOT/repo.git" ] || git init -q --bare -b main "$ROOT/repo.git"
install -m 755 "$HOOK_SRC" "$ROOT/repo.git/hooks/post-receive"

if [ -e "$WEB" ] && [ ! -L "$WEB" ]; then
  echo "$WEB 已存在且不是软链接，停止。" >&2
  exit 1
fi
ln -sfn "$ROOT/current" "$WEB"
echo "ok: $ROOT/repo.git 已就绪，$WEB -> $ROOT/current"
