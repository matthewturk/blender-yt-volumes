#!/usr/bin/env bash
# Check out the yt_idv this repo needs into ./yt_idv: upstream main at a pinned
# commit plus patches/yt_idv/yt_idv-780f233-integration.patch (open PRs
# 284-288 merged, and two small fixes). Then install it with
#
#     uv sync --group ytidv
#
# Re-running replaces ./yt_idv. See patches/yt_idv/README.md.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
commit=780f23307a2e76b96726130ef2b07be8595c9327
patch="$repo_root/patches/yt_idv/yt_idv-780f233-integration.patch"
target="$repo_root/yt_idv"

if [ -e "$target" ]; then
    echo "removing existing $target"
    rm -rf "$target"
fi
git clone --quiet https://github.com/yt-project/yt_idv.git "$target"
git -C "$target" checkout --quiet -b cycles-yt "$commit"
git -C "$target" apply "$patch"
git -C "$target" add -A
git -C "$target" -c user.name=cycles-yt -c user.email=cycles-yt@localhost \
    commit --quiet -m "yt_idv for cycles-yt: PRs 284-288 and two fixes"
echo "yt_idv ready in $target ($(git -C "$target" log --oneline -1))"
