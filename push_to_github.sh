#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# STEP 1: Create a new GitHub repo
#   Go to https://github.com/new
#   Name it: deepfake-voice-detector
#   Leave it empty (no README/gitignore)
#   Copy the SSH/HTTPS URL shown
#
# STEP 2: Run this script from the project root
# ─────────────────────────────────────────────────────────────

set -e

REPO_URL="${1:-https://github.com/YOUR_USERNAME/deepfake-voice-detector.git}"

echo "➜ Initialising git repository…"
git init
git add .
git commit -m "feat: initial commit — VoiceShield deepfake voice detector"

echo "➜ Pushing to GitHub ($REPO_URL)…"
git branch -M main
git remote add origin "$REPO_URL"
git push -u origin main

echo ""
echo "✅ Done! Your repository is live at:"
echo "   ${REPO_URL%.git}"
