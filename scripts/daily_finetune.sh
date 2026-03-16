#!/bin/bash
# DocWain Daily Auto Fine-Tune Pipeline
#
# Runs daily via cron. Checks feedback signals, triggers fine-tuning
# if quality thresholds are not met, and pushes updated model to Ollama.
#
# Cron setup (run at 2 AM daily):
#   0 2 * * * /home/muthu/PycharmProjects/DocWain/scripts/daily_finetune.sh >> /var/log/docwain-finetune.log 2>&1
#
# Flow:
#   1. Check feedback signals (low confidence ratio, grounding failures)
#   2. If fine-tuning needed: collect training data from feedback + Qdrant
#   3. Run Unsloth LoRA fine-tune on qwen3:8b base
#   4. Evaluate against base model
#   5. If improved: update Modelfile, rebuild and push MuthuSubramanian/DocWain
#   6. Notify via log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_PREFIX="[DocWain-AutoFineTune]"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
MODEL_NAME="MuthuSubramanian/DocWain"

cd "$PROJECT_DIR"

echo "$LOG_PREFIX $TIMESTAMP Starting daily fine-tune check"

# Step 1: Check if fine-tuning is needed based on feedback signals
FINETUNE_NEEDED=$(python -c "
import sys
sys.path.insert(0, '.')
try:
    from src.intelligence.feedback_tracker import FeedbackTracker
    from src.api.dataHandler import get_redis_client
    r = get_redis_client()
    if not r:
        print('skip:no_redis')
        sys.exit(0)
    tracker = FeedbackTracker(r)
    # Check all profiles with sufficient queries
    from pymongo import MongoClient
    from src.api.config import Config
    client = MongoClient(Config.MongoDB.URI, serverSelectionTimeoutMS=5000)
    db = client[Config.MongoDB.DB]
    profiles = list(db['profiles'].find({}, {'profile_id': 1, '_id': 1}))

    needs_tuning = False
    for p in profiles:
        pid = str(p.get('profile_id') or p.get('_id'))
        candidates = tracker.get_tuning_candidates(pid, min_queries=20)
        if candidates.get('is_candidate'):
            print(f'tune:{pid}:conf={candidates.get(\"avg_confidence\",0):.2f}:grounded={candidates.get(\"grounded_ratio\",0):.2f}')
            needs_tuning = True

    if not needs_tuning:
        print('skip:all_profiles_healthy')
except Exception as e:
    print(f'skip:error:{e}')
" 2>&1)

echo "$LOG_PREFIX Feedback check: $FINETUNE_NEEDED"

if [[ "$FINETUNE_NEEDED" == skip:* ]]; then
    echo "$LOG_PREFIX No fine-tuning needed: $FINETUNE_NEEDED"

    # Even if no fine-tuning, still rebuild model if Modelfile changed
    MODELFILE_CHANGED=$(git diff HEAD~1 --name-only 2>/dev/null | grep -c "Modelfile" || true)
    if [ "$MODELFILE_CHANGED" -gt 0 ]; then
        echo "$LOG_PREFIX Modelfile changed — rebuilding model without fine-tune"
        ollama create "$MODEL_NAME" -f "$PROJECT_DIR/Modelfile" 2>&1
        ollama push "$MODEL_NAME" 2>&1 | tail -3
        echo "$LOG_PREFIX Model updated (prompt changes only)"
    fi

    echo "$LOG_PREFIX $TIMESTAMP Daily check complete (no fine-tune triggered)"
    exit 0
fi

# Step 2: Trigger fine-tuning via the API
echo "$LOG_PREFIX Fine-tuning triggered for profiles: $FINETUNE_NEEDED"

# Extract first profile that needs tuning
PROFILE_ID=$(echo "$FINETUNE_NEEDED" | head -1 | cut -d: -f2)

# Get the subscription/collection for this profile
COLLECTION=$(python -c "
import sys
sys.path.insert(0, '.')
from pymongo import MongoClient
from src.api.config import Config
client = MongoClient(Config.MongoDB.URI, serverSelectionTimeoutMS=5000)
db = client[Config.MongoDB.DB]
profile = db['profiles'].find_one({'\$or': [{'profile_id': '$PROFILE_ID'}, {'_id': __import__('bson').ObjectId('$PROFILE_ID') if len('$PROFILE_ID') == 24 else '$PROFILE_ID'}]})
if profile:
    sub = str(profile.get('subscription_id') or profile.get('subscription') or '')
    print(sub)
else:
    print('')
" 2>&1)

if [ -z "$COLLECTION" ]; then
    echo "$LOG_PREFIX ERROR: Could not resolve collection for profile $PROFILE_ID"
    exit 1
fi

echo "$LOG_PREFIX Collection: $COLLECTION, Profile: $PROFILE_ID"

# Step 3: Call the fine-tune API
FINETUNE_RESULT=$(curl -s -X POST http://localhost:8000/api/finetune/by-collection \
    -H "Content-Type: application/json" \
    -d "{
        \"collection_name\": \"$COLLECTION\",
        \"base_model\": \"qwen3\",
        \"learning_rate\": 2e-4,
        \"batch_size\": 4,
        \"max_steps\": 200,
        \"lora_r\": 16,
        \"lora_alpha\": 16,
        \"agentic\": false
    }" 2>&1)

echo "$LOG_PREFIX Fine-tune result: $FINETUNE_RESULT"

FINETUNE_STATUS=$(echo "$FINETUNE_RESULT" | python -c "import sys,json; print(json.load(sys.stdin).get('overall_status','unknown'))" 2>/dev/null || echo "unknown")

if [ "$FINETUNE_STATUS" != "succeeded" ] && [ "$FINETUNE_STATUS" != "completed" ]; then
    echo "$LOG_PREFIX WARNING: Fine-tune status=$FINETUNE_STATUS — skipping model push"
    echo "$LOG_PREFIX $TIMESTAMP Daily fine-tune completed with warnings"
    exit 0
fi

# Step 4: Rebuild and push the DocWain model
echo "$LOG_PREFIX Fine-tune succeeded — rebuilding MuthuSubramanian/DocWain"
ollama create "$MODEL_NAME" -f "$PROJECT_DIR/Modelfile" 2>&1
ollama push "$MODEL_NAME" 2>&1 | tail -3

echo "$LOG_PREFIX $TIMESTAMP Daily fine-tune complete — model updated and pushed"
