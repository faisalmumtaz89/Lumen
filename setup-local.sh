#!/bin/bash
# ============================================================
#  Local -> Azure ML Compute Setup
#  Configures SSH multiplexing, sync helpers, and a remote-exec
#  workflow note for an Azure ML compute instance.
#  Run this ONCE on your laptop in your project directory.
# ============================================================
#  Usage:
#    ./setup-local.sh \
#        --host <public-ip-or-hostname> \
#        --port <ssh-port> \
#        --user <azureuser> \
#        --key  <path-to-pem> \
#        --alias <ssh-alias>      # optional, default: azml
#
#  Or set the same as env vars: REMOTE_HOST, REMOTE_PORT, REMOTE_USER,
#  IDENTITY_FILE, SSH_ALIAS.
# ============================================================

set -euo pipefail

# --- Defaults ---------------------------------------------------------------
REMOTE_USER="${REMOTE_USER:-azureuser}"
REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_PORT="${REMOTE_PORT:-}"
IDENTITY_FILE="${IDENTITY_FILE:-}"
SSH_ALIAS="${SSH_ALIAS:-azml}"

# --- Parse flags ------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --host)  REMOTE_HOST="$2"; shift 2;;
        --port)  REMOTE_PORT="$2"; shift 2;;
        --user)  REMOTE_USER="$2"; shift 2;;
        --key)   IDENTITY_FILE="$2"; shift 2;;
        --alias) SSH_ALIAS="$2"; shift 2;;
        -h|--help)
            sed -n '2,18p' "$0"; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

# --- Validate ---------------------------------------------------------------
missing=()
[ -z "$REMOTE_HOST"   ] && missing+=("--host")
[ -z "$REMOTE_PORT"   ] && missing+=("--port")
[ -z "$IDENTITY_FILE" ] && missing+=("--key")
if [ ${#missing[@]} -gt 0 ]; then
    echo "ERROR: missing required args: ${missing[*]}" >&2
    echo "Run with --help for usage." >&2
    exit 1
fi

# Expand ~ in IDENTITY_FILE
IDENTITY_FILE="${IDENTITY_FILE/#\~/$HOME}"
if [ ! -f "$IDENTITY_FILE" ]; then
    echo "ERROR: identity file not found: $IDENTITY_FILE" >&2
    exit 1
fi
chmod 600 "$IDENTITY_FILE"
echo "==> Set permissions 600 on $IDENTITY_FILE"

# --- 1. SSH config with connection multiplexing ----------------------------
mkdir -p ~/.ssh/cm && chmod 700 ~/.ssh/cm
touch ~/.ssh/config && chmod 600 ~/.ssh/config

# Strip any previous block for this alias, then append fresh
python3 - "$SSH_ALIAS" <<'PY'
import re, sys, pathlib
alias = sys.argv[1]
p = pathlib.Path.home() / ".ssh/config"
text = p.read_text() if p.exists() else ""
text = re.sub(rf"(?ms)^Host {re.escape(alias)}\b.*?(?=^Host |\Z)", "", text).rstrip() + "\n"
p.write_text(text)
PY

cat >> ~/.ssh/config <<EOF

Host $SSH_ALIAS
    HostName $REMOTE_HOST
    User $REMOTE_USER
    Port $REMOTE_PORT
    IdentityFile $IDENTITY_FILE
    IdentitiesOnly yes
    ServerAliveInterval 60
    ServerAliveCountMax 3
    ControlMaster auto
    ControlPath ~/.ssh/cm/%r@%h:%p
    ControlPersist 10m
    TCPKeepAlive yes
    StrictHostKeyChecking accept-new
EOF
echo "==> Added '$SSH_ALIAS' to ~/.ssh/config"

# --- 2. Test the connection -------------------------------------------------
echo "==> Testing SSH connection..."
if ssh -o BatchMode=yes -o ConnectTimeout=15 "$SSH_ALIAS" 'echo OK && hostname'; then
    echo "==> SSH works."
else
    echo ""
    echo "==> SSH failed. Common causes:" >&2
    echo "    - Compute instance is stopped in Azure ML Studio" >&2
    echo "    - Wrong host/port (check Studio -> Compute -> your instance)" >&2
    echo "    - .pem isn't the one registered with the compute instance" >&2
    echo "    - Your IP is blocked by an NSG rule" >&2
    exit 1
fi

# --- 3. Sync helpers --------------------------------------------------------
cat > ./sync-up.sh <<EOF
#!/bin/bash
rsync -avz --delete \\
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \\
  --exclude '.venv' --exclude 'node_modules' \\
  --exclude 'checkpoints/' --exclude 'data/' --exclude 'logs/' \\
  --exclude '.DS_Store' \\
  ./ $SSH_ALIAS:~/project/
EOF
chmod +x ./sync-up.sh

cat > ./pull-artifacts.sh <<EOF
#!/bin/bash
mkdir -p ./checkpoints ./logs
rsync -avz $SSH_ALIAS:~/project/checkpoints/ ./checkpoints/
rsync -avz $SSH_ALIAS:~/project/logs/ ./logs/
EOF
chmod +x ./pull-artifacts.sh

echo "==> Wrote sync-up.sh and pull-artifacts.sh"

# --- 4. CLAUDE.md -----------------------------------------------------------
MARKER="<!-- azml-remote-exec -->"
if [ -f CLAUDE.md ] && grep -q "$MARKER" CLAUDE.md; then
    echo "==> CLAUDE.md already has remote-exec block; leaving it alone."
else
    cat >> CLAUDE.md <<EOF

<!-- azml-remote-exec -->
# Remote Azure ML compute execution

This project runs heavy compute on a remote Azure ML instance reachable
as SSH host \`$SSH_ALIAS\`. **Default to running training/GPU/heavy commands
on the remote, not locally.** The local machine is for editing only.

## Workflow

1. Edit code locally as normal.
2. Before running anything heavy, sync: \`./sync-up.sh\`
3. Kick off a job via the remote helper:
   \`ssh $SSH_ALIAS '~/project/run.sh "python train.py --config configs/run1.yaml"'\`
   Runs inside a persistent tmux session that survives network drops.
   Returns the path to a timestamped log file.
4. Check progress: \`ssh $SSH_ALIAS '~/project/status.sh 120'\` (live tmux pane)
   or \`ssh $SSH_ALIAS '~/project/latest-log.sh 200'\` (newest log file).
5. Stop a running command (keeps session alive): \`ssh $SSH_ALIAS '~/project/stop.sh'\`
6. GPU health: \`ssh $SSH_ALIAS '~/project/gpu.sh'\`
7. Pull checkpoints/logs back: \`./pull-artifacts.sh\`

## Rules

- Never run training, GPU, or heavy CPU commands locally — always ssh.
- Never \`pip install\` locally for remote deps; install on the remote via
  \`ssh $SSH_ALIAS 'cd ~/project && pip install -r requirements.txt'\`.
- Datasets live at \`~/project/data/\` on the remote. Don't sync large data
  down; \`sync-up.sh\` already excludes \`data/\`, \`checkpoints/\`, \`logs/\`.
- Long runs (>5 min) must go through \`run.sh\` so they survive disconnects.
  Short one-offs (\`nvidia-smi\`, \`ls\`, \`cat\`) can just be \`ssh $SSH_ALIAS '...'\`.
- If \`ssh $SSH_ALIAS\` fails, the VM is probably stopped — tell the user, don't retry.

## Quick reference

| Goal                    | Command                                              |
|-------------------------|------------------------------------------------------|
| Sync code up            | \`./sync-up.sh\`                                     |
| Start a run             | \`ssh $SSH_ALIAS '~/project/run.sh "<cmd>"'\`        |
| Peek at live output     | \`ssh $SSH_ALIAS '~/project/status.sh 100'\`         |
| Tail newest log         | \`ssh $SSH_ALIAS '~/project/latest-log.sh 200'\`     |
| Stop current run        | \`ssh $SSH_ALIAS '~/project/stop.sh'\`               |
| GPU stats               | \`ssh $SSH_ALIAS '~/project/gpu.sh'\`                |
| Pull checkpoints/logs   | \`./pull-artifacts.sh\`                              |
| One-off remote shell    | \`ssh $SSH_ALIAS '<cmd>'\`                           |
<!-- /azml-remote-exec -->
EOF
    echo "==> Wrote CLAUDE.md (remote exec instructions appended)"
fi

echo ""
echo "============================================================"
echo "  Local setup done."
echo ""
echo "  Try it:  ssh $SSH_ALIAS '~/project/gpu.sh'"
echo ""
echo "  Then launch your editor/agent in this directory"
echo "  and start a run via the helpers above."
echo "============================================================"
