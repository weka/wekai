package llm

import (
	"testing"
)

func TestParseDynamicModel_SingleEndpoint(t *testing.T) {
	cfg, err := ParseDynamicModel("dynamic/http://localhost:8000/v1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.BaseURL != "http://localhost:8000/v1/" {
		t.Errorf("BaseURL = %q, want %q", cfg.BaseURL, "http://localhost:8000/v1/")
	}
	if len(cfg.BaseURLs) != 1 {
		t.Fatalf("len(BaseURLs) = %d, want 1", len(cfg.BaseURLs))
	}
	if cfg.BaseURLs[0] != cfg.BaseURL {
		t.Errorf("BaseURLs[0] = %q, want %q", cfg.BaseURLs[0], cfg.BaseURL)
	}
}

func TestParseDynamicModel_SingleEndpointWithParams(t *testing.T) {
	cfg, err := ParseDynamicModel("dynamic/http://localhost:8000/v1,type=openai,max_tokens=128")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.BaseURL != "http://localhost:8000/v1/" {
		t.Errorf("BaseURL = %q, want %q", cfg.BaseURL, "http://localhost:8000/v1/")
	}
	if cfg.Type != "openai" {
		t.Errorf("Type = %q, want %q", cfg.Type, "openai")
	}
	if cfg.MaxTokens != 128 {
		t.Errorf("MaxTokens = %d, want 128", cfg.MaxTokens)
	}
}

func TestParseDynamicModel_MultipleEndpoints(t *testing.T) {
	cfg, err := ParseDynamicModel("dynamic/http://ep1/v1|http://ep2/v1|http://ep3/v1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(cfg.BaseURLs) != 3 {
		t.Fatalf("len(BaseURLs) = %d, want 3", len(cfg.BaseURLs))
	}
	// BaseURL must be the first endpoint
	if cfg.BaseURL != "http://ep1/v1/" {
		t.Errorf("BaseURL = %q, want %q", cfg.BaseURL, "http://ep1/v1/")
	}
	if cfg.BaseURLs[0] != "http://ep1/v1/" {
		t.Errorf("BaseURLs[0] = %q, want %q", cfg.BaseURLs[0], "http://ep1/v1/")
	}
	if cfg.BaseURLs[1] != "http://ep2/v1/" {
		t.Errorf("BaseURLs[1] = %q, want %q", cfg.BaseURLs[1], "http://ep2/v1/")
	}
	if cfg.BaseURLs[2] != "http://ep3/v1/" {
		t.Errorf("BaseURLs[2] = %q, want %q", cfg.BaseURLs[2], "http://ep3/v1/")
	}
}

func TestParseDynamicModel_MultipleEndpointsWithParams(t *testing.T) {
	cfg, err := ParseDynamicModel("dynamic/http://ep1/v1|http://ep2/v1,type=openai,max_tokens=128")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(cfg.BaseURLs) != 2 {
		t.Fatalf("len(BaseURLs) = %d, want 2", len(cfg.BaseURLs))
	}
	if cfg.BaseURL != "http://ep1/v1/" {
		t.Errorf("BaseURL = %q, want %q", cfg.BaseURL, "http://ep1/v1/")
	}
	if cfg.BaseURLs[1] != "http://ep2/v1/" {
		t.Errorf("BaseURLs[1] = %q, want %q", cfg.BaseURLs[1], "http://ep2/v1/")
	}
	if cfg.Type != "openai" {
		t.Errorf("Type = %q, want %q", cfg.Type, "openai")
	}
	if cfg.MaxTokens != 128 {
		t.Errorf("MaxTokens = %d, want 128", cfg.MaxTokens)
	}
}

func TestParseDynamicModel_TrailingSlashNormalization(t *testing.T) {
	// URLs with trailing slash already should not get double slash
	cfg, err := ParseDynamicModel("dynamic/http://ep1/v1/|http://ep2/v1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.BaseURLs[0] != "http://ep1/v1/" {
		t.Errorf("BaseURLs[0] = %q, want %q", cfg.BaseURLs[0], "http://ep1/v1/")
	}
	if cfg.BaseURLs[1] != "http://ep2/v1/" {
		t.Errorf("BaseURLs[1] = %q, want %q", cfg.BaseURLs[1], "http://ep2/v1/")
	}
}

func TestParseDynamicModel_WithThinking(t *testing.T) {
	cfg, err := ParseDynamicModel("dynamic/http://localhost:8000/v1,type=openai_vllm,thinking=on")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Thinking != "on" {
		t.Errorf("Thinking = %q, want %q", cfg.Thinking, "on")
	}
}

func TestParseDynamicModel_BaseURLIsAlwaysFirst(t *testing.T) {
	cfg, err := ParseDynamicModel("dynamic/http://first/v1|http://second/v1|http://third/v1,model=mymodel")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.BaseURL != cfg.BaseURLs[0] {
		t.Errorf("BaseURL %q != BaseURLs[0] %q", cfg.BaseURL, cfg.BaseURLs[0])
	}
	if cfg.BaseURL != "http://first/v1/" {
		t.Errorf("BaseURL = %q, want %q", cfg.BaseURL, "http://first/v1/")
	}
	if cfg.Model != "mymodel" {
		t.Errorf("Model = %q, want %q", cfg.Model, "mymodel")
	}
}
