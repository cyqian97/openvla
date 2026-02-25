#!/bin/bash

# ============================================================
# tmux-session.sh - Launch a 4-pane tmux session
# ============================================================
# Usage: bash tmux-session.sh
#
# Layout:
#  +-----------+-----------+
#  |  Pane 0   |  Pane 1   |
#  +-----------+-----------+
#  |  Pane 2   |  Pane 3   |
#  +-----------+-----------+
# ============================================================

SESSION_NAME="my-session"

# Kill existing session if it exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null

# Create a new session (detached)
tmux new-session -d -s "$SESSION_NAME"

# Split into 4 panes
tmux split-window -h -t "$SESSION_NAME"       # Split pane 0 vertically → creates pane 1
tmux split-window -v -t "$SESSION_NAME:0.0"   # Split pane 0 horizontally → creates pane 2
tmux split-window -v -t "$SESSION_NAME:0.1"   # Split pane 1 horizontally → creates pane 3

# ============================================================
# Send commands to each pane
# ============================================================

# Pane 0 (top-left): SSH into server 1
tmux send-keys -t "$SESSION_NAME:0.0" \
  'ssh netbot@192.168.1.6' Enter
tmux send-keys -t "$SESSION_NAME:0.1" \
  'ssh netbot@192.168.1.6' Enter
sleep 2
tmux send-keys -t "$SESSION_NAME:0.0" \
  'docker exec -it nuc-setup_nuc-1 bash' Enter
tmux send-keys -t "$SESSION_NAME:0.0" \
  'cd ~/fr3/libfranka/build/examples' Enter
# sleep 1
# tmux send-keys -t "$SESSION_NAME:0.0" \
#   'env POLICY_HZ=15 bash /app/droid/franka_direct/launch_server.sh' Enter
# # Pane 1 (top-right): SSH into server 2
# tmux send-keys -t "$SESSION_NAME:0.3" \
#   'ssh netbot@192.168.1.6 "cd ~/fr3/libfranka/build/examples"' Enter

# # Pane 2 (bottom-left): SSH into server 3
# tmux send-keys -t "$SESSION_NAME:0.2" \
#   'ssh user@192.168.1.12 "tail -f /var/log/syslog"' Enter

# # Pane 3 (bottom-right): Local work
# tmux send-keys -t "$SESSION_NAME:0.3" \
#   'echo "Ready for local commands"' Enter

# ============================================================
# Attach to the session
# ============================================================
tmux attach-session -t "$SESSION_NAME"