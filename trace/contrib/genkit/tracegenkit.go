// Package genkit provides OpenTelemetry middleware for tracing Genkit model calls.
//
// First, set up tracing with braintrust.New():
//
//	tp := trace.NewTracerProvider()
//	defer tp.Shutdown(context.Background())
//	otel.SetTracerProvider(tp)
//
//	bt, err := braintrust.New(tp,
//		braintrust.WithProject("my-project"),
//	)
//	if err != nil {
//		log.Fatal(err)
//	}
//
// Then add the middleware to your Genkit generate call:
//
//	g := genkit.Init(ctx,
//		genkit.WithPlugins(&googlegenai.GoogleAI{APIKey: os.Getenv("GOOGLE_API_KEY")}),
//		genkit.WithDefaultModel("googleai/gemini-2.5-flash"),
//	)
//
//	resp, err := genkit.Generate(ctx, g,
//		ai.WithPrompt("Hello!"),
//		ai.WithMiddleware(genkittrace.NewMiddleware()),
//	)
//	if err != nil {
//		log.Fatal(err)
//	}
//	_ = resp
package genkit

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"unicode"

	"github.com/firebase/genkit/go/ai"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/codes"
	oteltrace "go.opentelemetry.io/otel/trace"

	"github.com/braintrustdata/braintrust-sdk-go/logger"
	"github.com/braintrustdata/braintrust-sdk-go/trace/internal"
)

type contextKey string

const activeLLMSpanKey contextKey = "braintrust.genkit.active_llm_span"

// config holds middleware configuration.
type config struct {
	tracerProvider oteltrace.TracerProvider
	logger         logger.Logger
}

// Option configures Genkit middleware.
type Option func(*config)

// WithTracerProvider sets a custom TracerProvider for the middleware.
// If not provided, the global otel.GetTracerProvider() is used.
func WithTracerProvider(tp oteltrace.TracerProvider) Option {
	return func(c *config) {
		c.tracerProvider = tp
	}
}

// WithLogger sets a custom logger for the middleware.
// If not provided, logging is disabled.
func WithLogger(log logger.Logger) Option {
	return func(c *config) {
		c.logger = log
	}
}

func (c *config) tracer() oteltrace.Tracer {
	tp := c.tracerProvider
	if tp == nil {
		tp = otel.GetTracerProvider()
	}
	return tp.Tracer("braintrust")
}

// NewMiddleware creates a new OpenTelemetry tracing middleware for Genkit model execution.
// By default, it uses the global TracerProvider. You can customize this with options.
func NewMiddleware(opts ...Option) ai.ModelMiddleware {
	cfg := &config{}
	for _, opt := range opts {
		opt(cfg)
	}

	return func(next ai.ModelFunc) ai.ModelFunc {
		return func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			if ctx.Value(activeLLMSpanKey) != nil {
				return next(ctx, req, cb)
			}

			ctx, span := cfg.tracer().Start(ctx, "genkit.model.generate")
			defer span.End()
			ctx = context.WithValue(ctx, activeLLMSpanKey, true)

			setJSONAttr(cfg, span, "braintrust.span_attributes", map[string]string{
				"type": "llm",
			})
			if input := normalizeRequest(req); input != nil {
				setJSONAttr(cfg, span, "braintrust.input_json", input)
			}

			reqMetadata := requestMetadata(req)
			resp, err := next(ctx, req, cb)
			if err != nil {
				ensureDefaultProvider(reqMetadata)
				if len(reqMetadata) > 0 {
					setJSONAttr(cfg, span, "braintrust.metadata", reqMetadata)
				}
				span.RecordError(err)
				span.SetStatus(codes.Error, err.Error())
				return nil, err
			}

			if output := normalizeResponse(resp); output != nil {
				setJSONAttr(cfg, span, "braintrust.output_json", output)
			}
			if metrics := usageMetrics(resp); len(metrics) > 0 {
				setJSONAttr(cfg, span, "braintrust.metrics", metrics)
			}

			metadata := reqMetadata
			for k, v := range responseMetadata(resp) {
				metadata[k] = v
			}
			ensureDefaultProvider(metadata)
			if len(metadata) > 0 {
				setJSONAttr(cfg, span, "braintrust.metadata", metadata)
			}

			return resp, nil
		}
	}
}

func setJSONAttr(cfg *config, span oteltrace.Span, key string, value any) {
	if err := internal.SetJSONAttr(span, key, value); err != nil {
		span.RecordError(err)
		if cfg.logger != nil {
			cfg.logger.Warn("failed to set Genkit tracing attribute", "key", key, "error", err)
		}
	}
}

func normalizeRequest(req *ai.ModelRequest) map[string]any {
	if req == nil {
		return nil
	}

	input := map[string]any{}
	if len(req.Messages) > 0 {
		input["messages"] = req.Messages
	}
	if len(req.Docs) > 0 {
		input["docs"] = req.Docs
	}
	if len(req.Tools) > 0 {
		input["tools"] = req.Tools
	}
	if req.ToolChoice != "" {
		input["tool_choice"] = req.ToolChoice
	}
	if req.Output != nil {
		input["output"] = req.Output
	}
	if req.Config != nil {
		input["config"] = req.Config
	}
	if len(input) == 0 {
		return nil
	}
	return input
}

func normalizeResponse(resp *ai.ModelResponse) map[string]any {
	if resp == nil {
		return nil
	}

	output := map[string]any{}
	if resp.Message != nil {
		output["message"] = resp.Message
	}
	if resp.FinishReason != "" {
		output["finish_reason"] = resp.FinishReason
	}
	if resp.FinishMessage != "" {
		output["finish_message"] = resp.FinishMessage
	}
	if resp.Operation != nil {
		output["operation"] = resp.Operation
	}
	if len(output) == 0 {
		return nil
	}
	return output
}

func requestMetadata(req *ai.ModelRequest) map[string]any {
	metadata := map[string]any{}
	if req == nil {
		return metadata
	}

	if system := systemPrompt(req.Messages); system != "" {
		metadata["system"] = system
	}
	if req.ToolChoice != "" {
		metadata["tool_choice"] = req.ToolChoice
	}
	if len(req.Tools) > 0 {
		metadata["tools"] = req.Tools
	}
	if req.Output != nil {
		metadata["output"] = req.Output
		if req.Output.Format != "" {
			metadata["response_format"] = req.Output.Format
		}
		if req.Output.ContentType != "" {
			metadata["content_type"] = req.Output.ContentType
		}
		if req.Output.Schema != nil {
			metadata["output_schema"] = req.Output.Schema
		}
		if req.Output.Constrained {
			metadata["output_constrained"] = true
		}
	}

	for k, v := range configMetadata(req.Config) {
		metadata[k] = v
	}

	return metadata
}

func responseMetadata(resp *ai.ModelResponse) map[string]any {
	metadata := map[string]any{}
	if resp == nil {
		return metadata
	}

	if resp.LatencyMs > 0 {
		metadata["latency_ms"] = resp.LatencyMs
	}

	custom := toMap(resp.Custom)
	if model, ok := stringValue(custom["model"]); ok {
		metadata["model"] = model
		if provider, ok := providerFromModel(model); ok {
			metadata["provider"] = provider
		}
	}
	if provider, ok := stringValue(custom["provider"]); ok {
		metadata["provider"] = provider
	}
	if id, ok := stringValue(custom["id"]); ok {
		metadata["id"] = id
	}
	if fingerprint, ok := stringValue(custom["systemFingerprint"]); ok {
		metadata["system_fingerprint"] = fingerprint
	}

	return metadata
}

func configMetadata(config any) map[string]any {
	metadata := map[string]any{}
	if config == nil {
		return metadata
	}

	if provider, ok := providerFromConfig(config); ok {
		metadata["provider"] = provider
	}

	cfgMap := toMap(config)
	if len(cfgMap) == 0 {
		return metadata
	}

	metadata["config"] = cfgMap

	if model, ok := firstString(cfgMap, "model", "model_name", "modelName"); ok {
		metadata["model"] = model
		if provider, ok := providerFromModel(model); ok {
			metadata["provider"] = provider
		}
	}
	if temp, ok := firstValue(cfgMap, "temperature", "Temperature"); ok {
		metadata["temperature"] = temp
	}
	if topP, ok := firstValue(cfgMap, "top_p", "topP", "TopP"); ok {
		metadata["top_p"] = topP
	}
	if topK, ok := firstValue(cfgMap, "top_k", "topK", "TopK"); ok {
		metadata["top_k"] = topK
	}
	if maxTokens, ok := firstValue(cfgMap, "max_output_tokens", "maxOutputTokens", "max_completion_tokens", "maxCompletionTokens", "max_tokens", "maxTokens"); ok {
		metadata["max_output_tokens"] = maxTokens
	}
	if stop, ok := firstValue(cfgMap, "stop_sequences", "stopSequences", "stop", "Stop"); ok {
		metadata["stop_sequences"] = stop
	}
	if responseFormat, ok := firstValue(cfgMap, "response_format", "responseFormat"); ok {
		metadata["response_format"] = responseFormat
	}
	if version, ok := firstValue(cfgMap, "version", "Version"); ok {
		metadata["version"] = version
	}

	return metadata
}

func usageMetrics(resp *ai.ModelResponse) map[string]any {
	if resp == nil || resp.Usage == nil {
		return nil
	}

	metrics := map[string]any{}
	usage := resp.Usage

	if usage.InputTokens > 0 {
		metrics["prompt_tokens"] = int64(usage.InputTokens)
	}
	if usage.OutputTokens > 0 {
		metrics["completion_tokens"] = int64(usage.OutputTokens)
	}
	if usage.TotalTokens > 0 {
		metrics["tokens"] = int64(usage.TotalTokens)
	}
	if usage.CachedContentTokens > 0 {
		metrics["prompt_cached_tokens"] = int64(usage.CachedContentTokens)
	}
	if usage.ThoughtsTokens > 0 {
		metrics["reasoning_tokens"] = int64(usage.ThoughtsTokens)
	}
	for key, value := range usage.Custom {
		if value == 0 {
			continue
		}
		metrics[snakeCase(key)] = value
	}

	if len(metrics) == 0 {
		return nil
	}
	if _, hasTotal := metrics["tokens"]; !hasTotal {
		prompt, okPrompt := metrics["prompt_tokens"].(int64)
		completion, okCompletion := metrics["completion_tokens"].(int64)
		if okPrompt && okCompletion {
			metrics["tokens"] = prompt + completion
		}
	}
	return metrics
}

func systemPrompt(messages []*ai.Message) string {
	var parts []string
	for _, msg := range messages {
		if msg == nil || msg.Role != ai.RoleSystem {
			continue
		}
		for _, part := range msg.Content {
			if part == nil {
				continue
			}
			if part.Text != "" {
				parts = append(parts, part.Text)
			}
		}
	}
	return strings.Join(parts, "\n")
}

func providerFromConfig(config any) (string, bool) {
	t := reflect.TypeOf(config)
	if t == nil {
		return "", false
	}
	for t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	pkg := t.PkgPath()
	switch {
	case strings.Contains(pkg, "openai-go"):
		return "openai", true
	case strings.Contains(pkg, "anthropic-sdk-go"):
		return "anthropic", true
	case strings.Contains(pkg, "google.golang.org/genai"):
		return "google", true
	default:
		return "", false
	}
}

func providerFromModel(model string) (string, bool) {
	if model == "" {
		return "", false
	}
	parts := strings.SplitN(model, "/", 2)
	if len(parts) != 2 || parts[0] == "" {
		return "", false
	}
	return parts[0], true
}

func toMap(v any) map[string]any {
	if v == nil {
		return nil
	}
	raw, err := json.Marshal(v)
	if err != nil {
		return nil
	}
	out := map[string]any{}
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil
	}
	return out
}

func firstValue(m map[string]any, keys ...string) (any, bool) {
	for _, key := range keys {
		value, ok := m[key]
		if ok && value != nil {
			return value, true
		}
	}
	return nil, false
}

func firstString(m map[string]any, keys ...string) (string, bool) {
	value, ok := firstValue(m, keys...)
	if !ok {
		return "", false
	}
	return stringValue(value)
}

func stringValue(v any) (string, bool) {
	s, ok := v.(string)
	if !ok || s == "" {
		return "", false
	}
	return s, true
}

func ensureDefaultProvider(metadata map[string]any) {
	if metadata == nil {
		return
	}
	if _, ok := metadata["provider"]; !ok {
		metadata["provider"] = "genkit"
	}
}

func snakeCase(s string) string {
	if s == "" {
		return s
	}
	var b strings.Builder
	lastLower := false
	for i, r := range s {
		if unicode.IsUpper(r) {
			if i > 0 && lastLower {
				b.WriteByte('_')
			}
			b.WriteRune(unicode.ToLower(r))
			lastLower = false
			continue
		}
		if r == '-' || r == ' ' {
			b.WriteByte('_')
			lastLower = false
			continue
		}
		b.WriteRune(r)
		lastLower = unicode.IsLetter(r) || unicode.IsDigit(r)
	}
	return b.String()
}
