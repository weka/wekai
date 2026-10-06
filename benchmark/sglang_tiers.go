package benchmark

import (
	"fmt"
	"sync/atomic"

	"github.com/weka/wekai/llm"
)

// applyCachedDetails records SGLang's per-request tier split on rec. A nil d
// (no breakdown reported) leaves the optional fields absent.
func (rec *requestDataRecord) applyCachedDetails(d *llm.CachedTokensDetails) {
	if d == nil {
		return
	}
	dev, host, sto := d.Device, d.Host, d.Storage
	rec.CachedDevice, rec.CachedHost, rec.CachedStorage = &dev, &host, &sto
	rec.CachedStorageBackend = d.StorageBackend
}

// sglangTierTotals accumulates per-request SGLang tier breakdowns across the
// run. Requests record concurrently, hence atomics.
type sglangTierTotals struct {
	reqs     atomic.Int64
	device   atomic.Int64
	host     atomic.Int64
	storage  atomic.Int64
	computed atomic.Int64
}

// add folds one request in. uncachedInput is the request's prompt tokens net of
// cached ones (what the server had to compute).
func (t *sglangTierTotals) add(d *llm.CachedTokensDetails, uncachedInput int) {
	if d == nil {
		return
	}
	t.reqs.Add(1)
	t.device.Add(int64(d.Device))
	t.host.Add(int64(d.Host))
	t.storage.Add(int64(d.Storage))
	if uncachedInput > 0 {
		t.computed.Add(int64(uncachedInput))
	}
}

type sglangTierSnapshot struct {
	reqs, device, host, storage, computed int64
}

func (t *sglangTierTotals) snapshot() sglangTierSnapshot {
	return sglangTierSnapshot{t.reqs.Load(), t.device.Load(), t.host.Load(), t.storage.Load(), t.computed.Load()}
}

// summaryLine is the end-of-run tier totals line, or "" when no request
// reported a breakdown.
func (s sglangTierSnapshot) summaryLine() string {
	if s.reqs == 0 {
		return ""
	}
	return fmt.Sprintf("sglang cache tiers: device=%s host=%s storage=%s computed=%s (%d reqs)",
		formatKiloInt(s.device), formatKiloInt(s.host), formatKiloInt(s.storage), formatKiloInt(s.computed), s.reqs)
}
