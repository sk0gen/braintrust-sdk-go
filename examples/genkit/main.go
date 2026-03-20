package main

import (
	"context"
	"fmt"
	"log"
	"os"

	braintrust "github.com/braintrustdata/braintrust-sdk-go"
	tracegenkit "github.com/braintrustdata/braintrust-sdk-go/trace/contrib/genkit"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

func main() {
	ctx := context.Background()

	tp := sdktrace.NewTracerProvider()
	defer func() {
		_ = tp.Shutdown(context.Background())
	}()
	otel.SetTracerProvider(tp)

	_, err := braintrust.New(tp, braintrust.WithProject("my-project"))
	if err != nil {
		log.Fatal(err)
	}

	g := genkit.Init(ctx,
		genkit.WithPlugins(&googlegenai.GoogleAI{
			APIKey: os.Getenv("GOOGLE_API_KEY"),
		}),
		genkit.WithDefaultModel("googleai/gemini-2.5-flash"),
	)

	resp, err := genkit.Generate(ctx, g,
		ai.WithPrompt("Say hello in one short sentence."),
		ai.WithMiddleware(tracegenkit.NewMiddleware(tracegenkit.WithTracerProvider(tp))),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(resp.Text())
}
