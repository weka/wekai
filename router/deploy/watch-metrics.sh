#!/usr/bin/env bash
# watch-metrics.sh — Poll the router's /metrics endpoint every INTERVAL
# seconds and print the fleet-load, route-decision, and cache-prediction
# observability: router_worker_load_{avg,max,min},
# router_route_decisions_total{decision="cache"|"load"|"other"}, and
# router_cache_prediction_{avg,max,min}.
#
# Usage:
#   router/deploy/watch-metrics.sh [METRICS_URL] [INTERVAL]
#
# METRICS_URL defaults to http://127.0.0.1:29000/metrics (the router's
# default -metrics-listen). INTERVAL defaults to 15 seconds.
#
# Example against a port-forwarded pod:
#   kubectl -n weka port-forward deploy/ofer-deepseek-v4-flash-weka-vllm-router 29000:29000 &
#   router/deploy/watch-metrics.sh http://127.0.0.1:29000/metrics 15
#
# router_route_decisions_total is a counter, so this prints both the
# cumulative total and the delta since the previous poll — the delta is
# what answers "how many of each kind just now", the cumulative is what
# Prometheus itself would show.

set -euo pipefail

METRICS_URL="${1:-http://127.0.0.1:29000/metrics}"
INTERVAL="${2:-15}"

command -v curl >/dev/null || { echo "ERROR: curl is required" >&2; exit 1; }

# metric_value NAME [LABEL_MATCH] < scrape
# LABEL_MATCH is an exact `{...}` suffix to match (e.g. '{decision="cache"}'),
# or empty for a bare gauge with no labels. Prints the value, or "n/a" if the
# series isn't present yet (e.g. no backends registered so far).
metric_value() {
    local name="$1" label="${2:-}"
    awk -v name="$name" -v label="$label" '
        $0 !~ /^#/ && index($0, name) == 1 {
            line = $0
            sub(name, "", line)
            if (label != "" && index(line, label) != 1) next
            n = split(line, parts, " ")
            print parts[n]
            found = 1
            exit
        }
        END { if (!found) print "n/a" }
    '
}

prev_cache=0
prev_load=0
prev_other=0
have_prev=0

echo "Polling $METRICS_URL every ${INTERVAL}s. Ctrl-C to stop."
echo

while true; do
    scrape="$(curl -sf "$METRICS_URL" || true)"
    ts="$(date '+%Y-%m-%d %H:%M:%S')"

    if [[ -z "$scrape" ]]; then
        echo "[$ts] ERROR: could not scrape $METRICS_URL"
        sleep "$INTERVAL"
        continue
    fi

    avg="$(metric_value router_worker_load_avg <<<"$scrape")"
    max="$(metric_value router_worker_load_max <<<"$scrape")"
    min="$(metric_value router_worker_load_min <<<"$scrape")"

    pred_avg="$(metric_value router_cache_prediction_avg <<<"$scrape")"
    pred_max="$(metric_value router_cache_prediction_max <<<"$scrape")"
    pred_min="$(metric_value router_cache_prediction_min <<<"$scrape")"

    cache="$(metric_value router_route_decisions_total '{decision="cache"}' <<<"$scrape")"
    load="$(metric_value router_route_decisions_total '{decision="load"}' <<<"$scrape")"
    other="$(metric_value router_route_decisions_total '{decision="other"}' <<<"$scrape")"

    if [[ "$have_prev" == 1 ]]; then
        d_cache="$(awk -v a="$cache" -v b="$prev_cache" 'BEGIN{ if (a=="n/a"||b=="n/a") print "n/a"; else printf "%+d", a-b }')"
        d_load="$(awk -v a="$load" -v b="$prev_load" 'BEGIN{ if (a=="n/a"||b=="n/a") print "n/a"; else printf "%+d", a-b }')"
        d_other="$(awk -v a="$other" -v b="$prev_other" 'BEGIN{ if (a=="n/a"||b=="n/a") print "n/a"; else printf "%+d", a-b }')"
    else
        d_cache="n/a"; d_load="n/a"; d_other="n/a"
    fi

    printf '[%s] load avg=%-6s max=%-6s min=%-6s | prediction avg=%-6s max=%-6s min=%-6s | decisions cache=%s(%s) load=%s(%s) other=%s(%s)\n' \
        "$ts" "$avg" "$max" "$min" \
        "$pred_avg" "$pred_max" "$pred_min" \
        "$cache" "$d_cache" "$load" "$d_load" "$other" "$d_other"

    prev_cache="$cache"; prev_load="$load"; prev_other="$other"
    have_prev=1

    sleep "$INTERVAL"
done
