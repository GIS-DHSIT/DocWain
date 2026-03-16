#!/bin/bash
# DocWain Daily Auto Fine-Tune — calls the API endpoints
#
# Cron: 0 2 * * * /home/muthu/PycharmProjects/DocWain/scripts/daily_finetune.sh >> /var/log/docwain-finetune.log 2>&1

set -euo pipefail
API="http://localhost:8000/api"
LOG="[DocWain-DailyFineTune]"

echo "$LOG $(date -u +%Y-%m-%dT%H:%M:%SZ) Starting"

# Check feedback signals
RESULT=$(curl -s "$API/model/feedback-check" 2>&1)
NEEDED=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('finetune_needed', False))" 2>/dev/null || echo "False")

echo "$LOG Feedback check: finetune_needed=$NEEDED"

if [ "$NEEDED" = "True" ]; then
    echo "$LOG Triggering auto fine-tune..."
    curl -s -X POST "$API/model/auto-finetune" -H "Content-Type: application/json" -d '{}' | python3 -m json.tool
else
    echo "$LOG No fine-tuning needed — checking Modelfile for prompt updates..."
    # Still push model if Modelfile changed (prompt improvements)
    cd /home/muthu/PycharmProjects/DocWain
    CHANGED=$(git diff HEAD~1 --name-only 2>/dev/null | grep -c "Modelfile" || true)
    if [ "$CHANGED" -gt 0 ]; then
        echo "$LOG Modelfile changed — triggering model update"
        curl -s -X POST "$API/model/update" -H "Content-Type: application/json" -d '{}' | python3 -m json.tool
    fi
fi

echo "$LOG $(date -u +%Y-%m-%dT%H:%M:%SZ) Complete"
