#!/bin/sh
# 发布当前提交到服务器的 main，由 post-receive 钩子完成上线。
# 可选参数：要发布的本地分支或提交；默认 HEAD。
# 需要已配置远端：git remote add deploy root@182.61.48.178:/srv/blackbox/repo.git
# 如果用密码登录，可以：SSHPASS=... GIT_SSH_COMMAND="sshpass -e ssh" sh tools/deploy.sh
set -eu
cd "$(dirname "$0")/.."
git push deploy "${1:-HEAD}:refs/heads/main"
