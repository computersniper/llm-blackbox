#!/bin/sh
# 发布：推送 main 到服务器上的 bare 仓库，由 post-receive 钩子完成上线。
# 需要已配置远端：git remote add deploy <用户>@<服务器>:/srv/blackbox/repo.git
# 如果用密码登录，可以：SSHPASS=... GIT_SSH_COMMAND="sshpass -e ssh" sh tools/deploy.sh
set -eu
cd "$(dirname "$0")/.."
git push deploy main
