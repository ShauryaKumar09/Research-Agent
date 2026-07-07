# Research Agent

An AI research agent built with LangGraph. Given a topic, it plans a set of
sub-questions, researches each one in parallel with web search, and merges the
results into a single markdown report (`report.md`).

It runs against a self-hosted LLM (`gpt-oss:20b` via Ollama) and uses Tavily
for web search. No hosted LLM API is required.

## How it works

```
                 +----------+
                 | planner  |  breaks the topic into 3-5 sub-questions
                 +----------+  (structured output, json_schema)
                      |
            Send() fan-out (one branch per sub-question)
              /       |       \
      +------------+------------+------------+
      | researcher | researcher | researcher |  Tavily search + summarize,
      +------------+------------+------------+  in parallel
              \       |       /
            findings merged via operator.add
                      |
                +-------------+
                | synthesizer |  writes the final markdown report
                +-------------+
                      |
                  report.md
```

### Architecture notes

- **Map-reduce with `Send()`.** The planner does not know ahead of time how
  many sub-questions there will be, so the graph cannot use a fixed number of
  edges. A conditional edge returns a list of `Send("researcher", ...)`
  objects, and LangGraph spawns one researcher branch per sub-question at
  runtime. Each branch receives only its own private state
  (`{"sub_question": ...}`).
- **Reducer state.** `findings` is declared as
  `Annotated[list[str], operator.add]`. Without a reducer, parallel branches
  writing to the same key would conflict; with `operator.add`, each branch's
  findings are appended, which is the reduce step.
- **Structured output.** The planner uses
  `.with_structured_output(ResearchPlan, method="json_schema")` — Ollama's
  native JSON schema mode — so the sub-question list comes back as a
  validated Pydantic object instead of free text that needs parsing. The
  `json_schema` method also avoids the tool-calling code path, which is not
  implemented for all Ollama models.
- **Local-first LLM.** All three node types call the same Ollama-served
  model. The server address comes from `OLLAMA_BASE_URL`, so the code never
  hardcodes a machine-specific address.

## Setup

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), a running
[Ollama](https://ollama.com/) server with `gpt-oss:20b` pulled, and a
[Tavily](https://tavily.com/) API key.

```sh
uv sync
cp .env.example .env
```

Then edit `.env`:

```
TAVILY_API_KEY=your-tavily-key-here
OLLAMA_BASE_URL=http://localhost:11434
```

Point `OLLAMA_BASE_URL` at wherever your Ollama server runs. On the server,
pull the model:

```sh
ollama pull gpt-oss:20b
```

## Usage

```sh
uv run main.py "how do solid state batteries work"
```

Prints the sub-questions it researched and writes the final report to
`report.md` (gitignored).

The same pipeline is also available as a notebook, `research_agent.ipynb`,
with a graph visualization and inline report rendering.

## Limitations

- A single Ollama server may serialize the "parallel" researcher requests
  depending on its `OLLAMA_NUM_PARALLEL` setting and available memory, so the
  fan-out is concurrent from the graph's point of view but not necessarily on
  the GPU.
- Requires a recent `langchain-ollama` for `method="json_schema"` structured
  output support.
- Report quality depends entirely on the top-3 Tavily results per
  sub-question; there is no citation verification or follow-up searching.
- No retries: if the model returns malformed structured output or a search
  fails, the run fails.
