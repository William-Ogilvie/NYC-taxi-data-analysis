#!/usr/bin/env bash
set -Eeuo pipefail
# E  : exit on any error in the script (safer than soldiering on)
# e  : (same as -E in many shells) propagate ERR traps in functions/subshells
# u  : treat unset variables as errors
# -o pipefail : fail a pipeline if *any* command fails (not just the last)

# --- Config ---
# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "Script directory: $SCRIPT_DIR"

# Get the project directory (one level above script directory)
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"
echo "Project directory: $PROJECT_DIR"

cd "$PROJECT_DIR" # ensure we are in the project directory

BUCKET="jfk-taxi-data-william-ogilvie" # S3 bucket to pull data from / push results to
IMAGE="jfk-taxi:cpu"          # docker image to use
USE_GPU=0                     # set to 1 to add --gpu all to docker run command
DATA_DIR="$PROJECT_DIR/data/processed" # data stored here locally
OUT_DIR="$PROJECT_DIR/data/saved_objects" # outputs stored here locally
RUN_ID="$(date +%F-%H%M%S)" # timestamp tag like 2025-09-23-154200 (unique run id)
LOG="$HOME/logs/exp-$RUN_ID.log" # log file on host

# --- Setup ---
# Create necessary directories if they don't exist
mkdir -p "$DATA_DIR" "$OUT_DIR" "$(dirname "$LOG")"

# Simple logging function
log(){ echo "[$(date '+%F %T')]" "$@" | tee -a "$LOG"; }
# adds timestamp to log messages, prints and appends to the log directory (tee -a does the appending, without -a it would overwrite) 


# upload the log to S3 if anything fails
trap 'log "ERROR on line $LINENO"; aws s3 cp "$LOG" "s3://$BUCKET/logs/exp-$RUN_ID.log" || true' ERR
# If any command errors (because of set -E…), this trap:
# 1) logs the line number,
# 2) uploads the current log to s3://$BUCKET/logs/..., and
# 3) `|| true` prevents the trap itself causing a second failure if the upload fails.

# Load the data from the S3 bucket
log "Syncing data from S3 -> $DATA_DIR ..."
aws s3 sync "s3://$BUCKET/data" "$DATA_DIR" --only-show-errors

log "Starting container run..."

# If USE_GPU is 1 then GPU_FLAG is --gpus all otherwise it's empty
GPU_FLAG=$([ "$USE_GPU" -eq 1 ] && echo "--gpus all" || true)

# --- Run the hyperparameter tuning in a docker container ---
# Run the container, mount the project directory, pass the UIC/GID to match host user and run the hyperparam tuning script
docker run --rm \
  -u "$(id -u):$(id -g)" \
  $GPU_FLAG \
  --mount type=bind,src="$(pwd)",dst=/app \
  "$IMAGE" \
  bash -lc 'pip install . && python scripts/hyperparam_tuning.py' | tee -a "$LOG"
# normal docker run commands except:
# bash -lc '...' : run a login bash (-l) and execute the command (-c). the login helps ensure the env is activated properly
# | tee -a "$LOG": stream container stdout to the log and your terminal
# we also pip install . to get the jfk_taxi package inside the container, then we just run the hyperparam tuning script



# Save the hyperparameter tuning outputs to the bucket with the unique run id
log "Syncing outputs to s3://$BUCKET/outputs/$RUN_ID ..."
aws s3 sync "$OUT_DIR" "s3://$BUCKET/outputs/$RUN_ID" --only-show-errors | tee -a "$LOG"

# Log completion and location of logs, turn off the instance
log "DONE. Logs at $LOG"
log "Turning off instance..."
sudo shutdown -h now


# To actually run this script:
# 1) ssh into EC2 instance
# 2) ensure in home dir, then mkdir -p project logs, then cd project
# 3) git clone https://github.com/William-Ogilvie/NYC-taxi-data-analysis.git
# 4) we will then use tmux to run this script so it keeps running if we disconnect
# 5) sudo apt-get update && sudo apt-get install -y tmux
# install tmux (allows for multiple terminal sessions)
# 6) tmux new -s hyperparam_tuning
# new tmux session called hyperparam_tuning
# to check this has worked do tmux detach to detach from the session then do tmux ls to see active sessions
# tmux attach -t hyperparam_tuning to reattach
# 7) then inside the tmux session:
# 8) cd project/NYC-taxi-data-analysis/scripts 
# 9) bash hyperparam_tuning_bash.sh 2>&1 
# run the script, 2>&1 ensures both stdout and stderr are logged into the same stream (i.e. the terminal)

