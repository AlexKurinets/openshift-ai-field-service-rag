#!/usr/bin/env bash
# Brings up the Compose stack and runs every command sequence the README documents, in order.
set -euo pipefail
cd "$(dirname "$0")/.."

API="http://localhost:${API_PORT:-8170}"
TOKEN="${FA_API_TOKEN:-local-dev-token}"
NEW_IMAGE="oci://quay.example.com/fieldassist/granite-3.3-8b-instruct@sha256:5e2b8d1f4a7c0e3b6d9f2a5c8e1b4d7f0a3c6e9b2d5f8a1c4e7b0d3f6a9c2e5b"
fa() { docker compose exec -T gateway fieldassist "$@"; }
ask() {
  curl -fsS -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
    -d "{\"question\": \"$1\"}" "$API/v1/ask"
}
expect() {  # expect <label> <haystack> <needle>
  if ! grep -qF -- "$3" <<<"$2"; then echo "FAIL: $1: expected '$3' in: $2"; exit 1; fi
  echo "ok: $1"
}

echo "== stack"
docker compose up -d --build --wait
expect "readyz" "$(curl -fsS "$API/readyz")" '"ready":true'

echo "== answers"
expect "superseded torque comes from the bulletin" \
  "$(ask 'What torque do the airend coupling bolts on the KR-160 take?')" "105 Nm"
expect "fault code found by exact match" "$(ask 'What does fault F6580 mean?')" "emergency stop pressed"
expect "unknown model line refused" "$(ask 'How often should I change the oil on the TD-150?')" '"refused":true'
expect "index status" "$(curl -fsS -H "Authorization: Bearer $TOKEN" "$API/v1/index/status")" '"bulletin":8'

echo "== golden set, three configurations"
fa eval --config config/retrieval/production.yaml --out reports/production.json
fa eval --config config/retrieval/vector-only.yaml --out reports/vector-only.json
fa eval --config config/retrieval/no-revision-filter.yaml --out reports/no-revision-filter.json

echo "== gate"
for candidate in vector-only no-revision-filter; do
  if out=$(fa gate --baseline reports/production.json --candidate "reports/$candidate.json"); then
    echo "$out"; echo "FAIL: gate passed $candidate"; exit 1
  fi
  expect "gate blocks $candidate" "$out" "decision: BLOCK"
done

echo "== promotion"
fa promote --baseline reports/production.json --candidate reports/production.json >/dev/null
fa eval --config config/retrieval/production.yaml --model-image "$NEW_IMAGE" --out reports/candidate.json
fa baseline --environment prod --out reports/baseline.json
out=$(fa promote --baseline reports/baseline.json --candidate reports/candidate.json)
expect "promotion passes and rewrites the model image" "$out" "+      storageUri: $NEW_IMAGE"

echo "smoke passed"
