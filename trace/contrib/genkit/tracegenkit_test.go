package genkit

import (
	"context"
	"errors"
	"net"
	"net/http"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	compatopenai "github.com/firebase/genkit/go/plugins/compat_oai/openai"
	openaigo "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/codes"

	"github.com/braintrustdata/braintrust-sdk-go/internal/oteltest"
	"github.com/braintrustdata/braintrust-sdk-go/internal/vcr"
)

func TestMiddlewareTracesBasicRequestResponse(t *testing.T) {
	tp, exporter := oteltest.Setup(t)

	req := &ai.ModelRequest{
		Messages: []*ai.Message{
			ai.NewUserTextMessage("Hello from Genkit"),
		},
	}
	resp := &ai.ModelResponse{
		Message: ai.NewModelTextMessage("Hello from Braintrust"),
		Custom: map[string]any{
			"model": "gpt-4o-mini",
		},
	}

	_, err := executeMiddleware(context.Background(), NewMiddleware(WithTracerProvider(tp)), req, resp, nil)
	require.NoError(t, err)

	span := exporter.FlushOne()
	span.AssertNameIs("genkit.model.generate")
	span.AssertJSONAttrEquals("braintrust.span_attributes", map[string]any{
		"type": "llm",
	})

	input := span.Input().(map[string]any)
	messages := input["messages"].([]any)
	firstMessage := messages[0].(map[string]any)
	assert.Equal(t, "user", firstMessage["role"])

	output := span.Output().(map[string]any)
	message := output["message"].(map[string]any)
	content := message["content"].([]any)
	firstPart := content[0].(map[string]any)
	assert.Equal(t, "Hello from Braintrust", firstPart["text"])

	metadata := span.Metadata()
	assert.Equal(t, "gpt-4o-mini", metadata["model"])
	assert.Equal(t, "genkit", metadata["provider"])
}

func TestMiddlewareCapturesMetadataAndMetrics(t *testing.T) {
	tp, exporter := oteltest.Setup(t)

	req := &ai.ModelRequest{
		Config: &openaigo.ChatCompletionNewParams{
			Temperature:         openaigo.Float(0.2),
			TopP:                openaigo.Float(0.8),
			MaxCompletionTokens: openaigo.Int(32),
		},
		Messages: []*ai.Message{
			ai.NewSystemTextMessage("You are concise."),
			ai.NewUserTextMessage("Return JSON"),
		},
		Output: &ai.ModelOutputConfig{
			Format: "json",
			Schema: map[string]any{"type": "object"},
		},
		ToolChoice: ai.ToolChoiceRequired,
		Tools: []*ai.ToolDefinition{
			{
				Name:        "lookup_weather",
				Description: "Looks up weather",
				InputSchema: map[string]any{
					"type": "object",
				},
			},
		},
	}
	resp := &ai.ModelResponse{
		Message: ai.NewModelTextMessage(`{"city":"Paris"}`),
		Custom: map[string]any{
			"model":             "gpt-4o-mini",
			"systemFingerprint": "fp_123",
			"id":                "resp_123",
		},
		Usage: &ai.GenerationUsage{
			InputTokens:         11,
			OutputTokens:        7,
			TotalTokens:         18,
			CachedContentTokens: 3,
			ThoughtsTokens:      2,
			Custom: map[string]float64{
				"acceptedPredictionTokens": 5,
			},
		},
	}

	_, err := executeMiddleware(context.Background(), NewMiddleware(WithTracerProvider(tp)), req, resp, nil)
	require.NoError(t, err)

	span := exporter.FlushOne()
	metadata := span.Metadata()
	assert.Equal(t, "openai", metadata["provider"])
	assert.Equal(t, "gpt-4o-mini", metadata["model"])
	assert.Equal(t, "You are concise.", metadata["system"])
	assert.Equal(t, "json", metadata["response_format"])
	assert.Equal(t, "required", metadata["tool_choice"])
	assert.Equal(t, "fp_123", metadata["system_fingerprint"])
	assert.Equal(t, "resp_123", metadata["id"])
	assert.Equal(t, 0.2, metadata["temperature"])
	assert.Equal(t, 0.8, metadata["top_p"])
	assert.Equal(t, 32.0, metadata["max_output_tokens"])

	metrics := span.Metrics()
	assert.Equal(t, 11.0, metrics["prompt_tokens"])
	assert.Equal(t, 7.0, metrics["completion_tokens"])
	assert.Equal(t, 18.0, metrics["tokens"])
	assert.Equal(t, 3.0, metrics["prompt_cached_tokens"])
	assert.Equal(t, 2.0, metrics["reasoning_tokens"])
	assert.Equal(t, 5.0, metrics["accepted_prediction_tokens"])
}

func TestMiddlewareRecordsErrors(t *testing.T) {
	tp, exporter := oteltest.Setup(t)

	req := &ai.ModelRequest{
		Messages: []*ai.Message{
			ai.NewUserTextMessage("This will fail"),
		},
	}

	_, err := executeMiddleware(context.Background(), NewMiddleware(WithTracerProvider(tp)), req, nil, errors.New("genkit boom"))
	require.Error(t, err)

	span := exporter.FlushOne()
	assert.Equal(t, codes.Error, span.Status().Code)
	assert.Contains(t, span.Status().Description, "genkit boom")
	assert.True(t, span.HasAttr("braintrust.input_json"))
	assert.True(t, span.HasAttr("braintrust.metadata"))
	assert.False(t, span.HasAttr("braintrust.output_json"))
}

func TestMiddlewareHandlesNilData(t *testing.T) {
	tp, exporter := oteltest.Setup(t)

	_, err := executeMiddleware(context.Background(), NewMiddleware(WithTracerProvider(tp)), nil, &ai.ModelResponse{}, nil)
	require.NoError(t, err)

	span := exporter.FlushOne()
	span.AssertNameIs("genkit.model.generate")
	assert.True(t, span.HasAttr("braintrust.span_attributes"))
	assert.False(t, span.HasAttr("braintrust.input_json"))
	assert.True(t, span.HasAttr("braintrust.metadata"))
}

func TestMiddlewareIntegration(t *testing.T) {
	tp, exporter := oteltest.Setup(t)

	mode := vcr.GetVCRMode()
	if mode != vcr.ModeReplay {
		server := startFixedOpenAIServer(t)
		t.Cleanup(func() {
			_ = server.Close()
		})
	}

	ctx := context.Background()
	httpClient := vcr.NewHTTPClient(t)
	plugin := &compatopenai.OpenAI{
		APIKey: "dummy-openai-key",
		Opts: []option.RequestOption{
			option.WithBaseURL("http://127.0.0.1:38087/v1"),
			option.WithHTTPClient(httpClient),
		},
	}

	g := genkit.Init(ctx,
		genkit.WithPlugins(plugin),
		genkit.WithDefaultModel("openai/gpt-4o-mini"),
	)

	resp, err := genkit.Generate(ctx, g,
		ai.WithPrompt("What is the capital of France?"),
		ai.WithConfig(&openaigo.ChatCompletionNewParams{
			Model:               "gpt-4o-mini",
			Temperature:         openaigo.Float(0.2),
			MaxCompletionTokens: openaigo.Int(32),
		}),
		ai.WithMiddleware(NewMiddleware(WithTracerProvider(tp))),
	)
	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, "Paris", resp.Text())

	spans := exporter.Flush()
	var llmSpan *oteltest.Span
	for i := range spans {
		if spans[i].Name() == "genkit.model.generate" {
			llmSpan = &spans[i]
			break
		}
	}
	require.NotNil(t, llmSpan)

	input := llmSpan.Input().(map[string]any)
	messages := input["messages"].([]any)
	firstMessage := messages[0].(map[string]any)
	content := firstMessage["content"].([]any)
	assert.Equal(t, "What is the capital of France?", content[0].(map[string]any)["text"])

	output := llmSpan.Output().(map[string]any)
	message := output["message"].(map[string]any)
	outContent := message["content"].([]any)
	assert.Equal(t, "Paris", outContent[0].(map[string]any)["text"])

	metadata := llmSpan.Metadata()
	assert.Equal(t, "openai", metadata["provider"])
	assert.Equal(t, "gpt-4o-mini", metadata["model"])
	assert.Equal(t, 0.2, metadata["temperature"])
	assert.Equal(t, 32.0, metadata["max_output_tokens"])

	metrics := llmSpan.Metrics()
	assert.Equal(t, 11.0, metrics["prompt_tokens"])
	assert.Equal(t, 3.0, metrics["completion_tokens"])
	assert.Equal(t, 14.0, metrics["tokens"])
}

func executeMiddleware(
	ctx context.Context,
	mw ai.ModelMiddleware,
	req *ai.ModelRequest,
	resp *ai.ModelResponse,
	nextErr error,
) (*ai.ModelResponse, error) {
	fn := mw(func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return resp, nextErr
	})
	return fn(ctx, req, nil)
}

func startFixedOpenAIServer(t *testing.T) *http.Server {
	t.Helper()

	listener, err := net.Listen("tcp", "127.0.0.1:38087")
	require.NoError(t, err)

	server := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, "/v1/chat/completions", r.URL.Path)
			assert.Equal(t, "POST", r.Method)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
  "id": "chatcmpl_test_123",
  "object": "chat.completion",
  "created": 1742428800,
  "model": "gpt-4o-mini",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Paris"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 11,
    "completion_tokens": 3,
    "total_tokens": 14
  }
}`))
		}),
	}

	go func() {
		_ = server.Serve(listener)
	}()

	return server
}
