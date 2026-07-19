package benchmark

// Hermes-style conversation dataset loader. Reads parquet files from the
// lambda/hermes-agent-reasoning-traces HuggingFace dataset (glm-5.1 split),
// caches them under ~/.cache/hermes-demo, and decodes into []Conversation.
//
// Modelled after the hermes-dataset-convert utility (sibling project) — same
// HF-API endpoint, same on-disk cache layout, same Arrow decoding path.

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/parquet/file"
	"github.com/apache/arrow-go/v18/parquet/pqarrow"
)

// HermesTurn is one role-tagged message inside a conversation.
// From is one of: "system", "human", "gpt", "tool".
type HermesTurn struct {
	From  string
	Value string
}

// Conversation is one full multi-turn dialogue from the dataset.
type Conversation struct {
	ID    string
	Turns []HermesTurn
}

type datasetSpec struct {
	repo   string
	config string
	split  string
}

// Short-name → HF dataset location. Extend this map to support new datasets.
var datasetSpecs = map[string]datasetSpec{
	"hermes-lambda": {
		repo:   "lambda/hermes-agent-reasoning-traces",
		config: "glm-5.1",
		split:  "train",
	},
}

// LoadConversationDataset resolves a short dataset name, caches the underlying
// parquet files, and returns up to `limit` conversations in dataset order.
// Passing limit <= 0 returns every conversation in the split.
func LoadConversationDataset(ctx context.Context, shortName string, limit int) ([]Conversation, error) {
	spec, ok := datasetSpecs[shortName]
	if !ok {
		known := make([]string, 0, len(datasetSpecs))
		for k := range datasetSpecs {
			known = append(known, k)
		}
		sort.Strings(known)
		return nil, fmt.Errorf("unknown dataset %q (known: %v)", shortName, known)
	}

	urls, err := listParquetURLs(ctx, spec.repo, spec.config, spec.split)
	if err != nil {
		return nil, fmt.Errorf("list parquet URLs: %w", err)
	}

	cacheDir := filepath.Join(xdgCacheHome(), "hermes-demo", spec.repo, spec.config, spec.split)
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return nil, fmt.Errorf("mkdir cache: %w", err)
	}

	var out []Conversation
	for i, url := range urls {
		path := filepath.Join(cacheDir, fmt.Sprintf("%d.parquet", i))
		if err := ensureCached(ctx, url, path); err != nil {
			return nil, fmt.Errorf("cache %s: %w", url, err)
		}
		remaining := -1
		if limit > 0 {
			remaining = limit - len(out)
			if remaining <= 0 {
				break
			}
		}
		rows, err := readParquetRows(ctx, path, remaining)
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", path, err)
		}
		out = append(out, rows...)
		if limit > 0 && len(out) >= limit {
			out = out[:limit]
			break
		}
	}
	return out, nil
}

func listParquetURLs(ctx context.Context, repo, config, split string) ([]string, error) {
	req, err := http.NewRequestWithContext(ctx, "GET",
		"https://huggingface.co/api/datasets/"+repo+"/parquet", nil)
	if err != nil {
		return nil, err
	}
	if tok := os.Getenv("HF_TOKEN"); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("status %d: %s", resp.StatusCode, body)
	}
	var tree map[string]map[string][]string
	if err := json.NewDecoder(resp.Body).Decode(&tree); err != nil {
		return nil, err
	}
	urls := tree[config][split]
	if len(urls) == 0 {
		return nil, fmt.Errorf("no parquet files for %s/%s", config, split)
	}
	sort.Strings(urls)
	return urls, nil
}

func ensureCached(ctx context.Context, url, path string) error {
	if st, err := os.Stat(path); err == nil && st.Size() > 0 {
		return nil
	}
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return err
	}
	if tok := os.Getenv("HF_TOKEN"); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("download status %d: %s", resp.StatusCode, body)
	}
	tmp := path + ".part"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := io.Copy(f, resp.Body); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// readParquetRows streams parquet rows and decodes into Conversation.
// If limit > 0, stops after limit conversations.
func readParquetRows(ctx context.Context, path string, limit int) ([]Conversation, error) {
	pf, err := file.OpenParquetFile(path, false)
	if err != nil {
		return nil, err
	}
	defer pf.Close()

	arrowRdr, err := pqarrow.NewFileReader(pf,
		pqarrow.ArrowReadProperties{BatchSize: 64},
		memory.DefaultAllocator)
	if err != nil {
		return nil, err
	}
	rr, err := arrowRdr.GetRecordReader(ctx, nil, nil)
	if err != nil {
		return nil, err
	}
	defer rr.Release()

	var out []Conversation
	for rr.Next() {
		rec := rr.Record()
		rows := recordToConversations(rec)
		for _, r := range rows {
			out = append(out, r)
			if limit > 0 && len(out) >= limit {
				return out, nil
			}
		}
	}
	if err := rr.Err(); err != nil {
		return out, err
	}
	return out, nil
}

// recordToConversations decodes one Arrow record batch. Schema:
// id (utf8), conversations (list<struct<from:utf8, value:utf8>>).
func recordToConversations(rec arrow.Record) []Conversation {
	schema := rec.Schema()
	colIdx := func(name string) int {
		for i, f := range schema.Fields() {
			if f.Name == name {
				return i
			}
		}
		return -1
	}

	idCol := rec.Column(colIdx("id")).(*array.String)
	convCol := rec.Column(colIdx("conversations")).(*array.List)
	convStruct := convCol.ListValues().(*array.Struct)
	structType := convStruct.DataType().(*arrow.StructType)
	fromIdx, _ := structType.FieldIdx("from")
	valueIdx, _ := structType.FieldIdx("value")
	fromArr := convStruct.Field(fromIdx).(*array.String)
	valueArr := convStruct.Field(valueIdx).(*array.String)

	out := make([]Conversation, rec.NumRows())
	for i := int64(0); i < rec.NumRows(); i++ {
		start, end := convCol.ValueOffsets(int(i))
		turns := make([]HermesTurn, 0, end-start)
		for j := start; j < end; j++ {
			turns = append(turns, HermesTurn{
				From:  fromArr.Value(int(j)),
				Value: valueArr.Value(int(j)),
			})
		}
		out[i] = Conversation{
			ID:    idCol.Value(int(i)),
			Turns: turns,
		}
	}
	return out
}

func xdgCacheHome() string {
	if v := os.Getenv("XDG_CACHE_HOME"); v != "" {
		return v
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".cache")
}
