#!/usr/bin/env bash
# Colab bootstrap for Part 1 (PRIVATE repo — needs a GitHub token).
#   - clone (or pull) the private repo into the Colab workspace using $GITHUB_TOKEN
#   - install dependencies
#   - create the Google Drive results folders
#   - print the paths the notebook will use
#
# The notebook loads the token first (Colab userdata or hidden prompt), exports
# it as GITHUB_TOKEN, then fetches THIS script over an authenticated curl and
# runs it. The token is used only to clone, then scrubbed from the git remote so
# it is never written to disk in .git/config.
#
# Env overrides:
#   GITHUB_TOKEN  (required) GitHub PAT with read access to the private repo
#   GH_USER       GitHub owner   (default: silvergjeka22)
#   REPO          repo name      (default: nesy-highway-driving)
#   WORKDIR       clone target   (default: /content/nesy-highway-driving)
#   DRIVE_ROOT    Drive results root (default matches configs/highway.yaml)

set -euo pipefail

GH_USER="${GH_USER:-silvergjeka22}"
REPO="${REPO:-nesy-highway-driving}"
WORKDIR="${WORKDIR:-/content/${REPO}}"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/nesy-highway-driving}"

if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "ERROR: GITHUB_TOKEN is not set. Load the token in the notebook first." >&2
  exit 1
fi

AUTH_URL="https://${GH_USER}:${GITHUB_TOKEN}@github.com/${GH_USER}/${REPO}.git"
CLEAN_URL="https://github.com/${GH_USER}/${REPO}.git"

echo "==> Repo:       ${GH_USER}/${REPO} (private)"
echo "==> Workdir:    $WORKDIR"
echo "==> Drive root: $DRIVE_ROOT"

# 1. Clone or update the repo (token used only here).
if [ -d "$WORKDIR/.git" ]; then
  echo "==> Repo exists; pulling latest."
  git -C "$WORKDIR" remote set-url origin "$AUTH_URL"
  git -C "$WORKDIR" pull --ff-only
else
  echo "==> Cloning private repo."
  git clone "$AUTH_URL" "$WORKDIR"
fi

# Scrub the token from the stored remote so it never persists in .git/config.
git -C "$WORKDIR" remote set-url origin "$CLEAN_URL"

# 2. Install dependencies (xvfb enables headless .mp4 rendering on Colab —
#    without it, highway-env's pygame renderer segfaults the kernel during video).
echo "==> Installing xvfb (headless rendering)."
apt-get -qq update -y >/dev/null 2>&1 || true
if ! apt-get -qq install -y xvfb >/dev/null 2>&1; then
  echo "    WARNING: xvfb install failed — video rendering may produce blank frames." >&2
fi
echo "==> Installing requirements."
pip install -q -r "$WORKDIR/requirements.txt"

# Colab's SciPy/pandas are built for its stock NumPy (2.0.x). Keep NumPy in the
# compatible range WITHOUT -U (never bump Colab's stock build): <2.0 breaks those
# wheels ("dtype size changed"); >=2.3 breaks NumPy's own C-API ("_blas_supports_fpe").
echo "==> Keeping NumPy in the SciPy-compatible range (>=2.0,<2.3)."
pip install -q "numpy>=2.0,<2.3"

# 3. Create Drive results folders (mirrors paths in configs/highway.yaml).
echo "==> Creating Drive results folders."
for sub in checkpoints videos metrics metrics/curves; do
  mkdir -p "$DRIVE_ROOT/$sub"
done

echo "==> Done."
echo "    Add the repo to sys.path in Python:  import sys; sys.path.insert(0, '$WORKDIR')"
echo "    Results will mirror to:              $DRIVE_ROOT"
