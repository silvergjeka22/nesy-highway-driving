#!/usr/bin/env bash
# Colab bootstrap for Part 1.
#   - clone (or pull) the repo into the Colab workspace
#   - install dependencies
#   - create the Google Drive results folders
#   - print the paths the notebook will use
#
# Usage (from the notebook, after mounting Drive):
#   !bash bash/setup_colab.sh
# or, on a fresh runtime before the repo exists:
#   !bash <(curl -fsSL <raw-url>/bash/setup_colab.sh)
#
# Env overrides:
#   REPO_URL   git remote (default: this project's GitHub URL — edit below)
#   WORKDIR    where to clone (default: /content/nesy-highway-driving)
#   DRIVE_ROOT Drive results root (default matches configs/highway.yaml)

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/silvergjeka22/nesy-highway-driving.git}"
WORKDIR="${WORKDIR:-/content/nesy-highway-driving}"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/nesy-highway-driving}"

echo "==> Repo:       $REPO_URL"
echo "==> Workdir:    $WORKDIR"
echo "==> Drive root: $DRIVE_ROOT"

# 1. Clone or update the repo.
if [ -d "$WORKDIR/.git" ]; then
  echo "==> Repo exists; pulling latest."
  git -C "$WORKDIR" pull --ff-only
else
  echo "==> Cloning repo."
  git clone "$REPO_URL" "$WORKDIR"
fi

# 2. Install dependencies.
echo "==> Installing requirements."
pip install -q -r "$WORKDIR/requirements.txt"

# 3. Create Drive results folders (mirrors paths in configs/highway.yaml).
echo "==> Creating Drive results folders."
for sub in checkpoints videos metrics tb; do
  mkdir -p "$DRIVE_ROOT/$sub"
done

echo "==> Done."
echo "    Add the repo to sys.path in Python:  import sys; sys.path.insert(0, '$WORKDIR')"
echo "    Results will mirror to:              $DRIVE_ROOT"
