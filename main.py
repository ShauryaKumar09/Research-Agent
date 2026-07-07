"""AI research agent: planner -> parallel researchers via Send() -> synthesizer."""

import operator
import os
import sys
from typing import Annotated

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

# base_url comes from .env so the code never hardcodes a LAN IP
llm = init_chat_model(
    model="ollama:gpt-oss:20b",
    temperature=0.2,  # low temp = reliable planning/synthesis
    base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
)

search_tool = TavilySearch(max_results=3)


class ResearchState(TypedDict):
    topic: str
    sub_questions: list[str]
    findings: Annotated[list[str], operator.add]  # reducer merges parallel branches
    report: str


class ResearcherState(TypedDict):
    sub_question: str


class ResearchPlan(BaseModel):
    sub_questions: list[str] = Field(
        description="3-5 focused, independently-searchable sub-questions "
        "that together fully cover the research topic."
    )


def planner(state: ResearchState):
    """Break the topic into sub-questions using structured output."""
    planner_llm = llm.with_structured_output(ResearchPlan, method="json_schema")
    plan = planner_llm.invoke(
        f"You are a research planner. Break this topic into 3-5 focused "
        f"sub-questions that can each be answered with a web search:\n\n"
        f"Topic: {state['topic']}"
    )
    return {"sub_questions": plan.sub_questions}


def fan_out(state: ResearchState):
    """Send() one researcher per sub-question — they run in parallel."""
    return [Send("researcher", {"sub_question": q}) for q in state["sub_questions"]]


def researcher(state: ResearcherState):
    """Search the web for one sub-question and summarize the results."""
    question = state["sub_question"]
    results = search_tool.invoke({"query": question})

    summary = llm.invoke(
        f"Summarize these search results into 2-3 dense paragraphs that "
        f"answer the question. Cite source URLs inline.\n\n"
        f"Question: {question}\n\nSearch results:\n{results}"
    )
    return {"findings": [f"### {question}\n\n{summary.content}"]}


def synthesizer(state: ResearchState):
    """Merge all findings into one final markdown report."""
    findings_block = "\n\n---\n\n".join(state["findings"])
    report = llm.invoke(
        f"You are a research writer. Using ONLY the findings below, write a "
        f"well-organized markdown report on: {state['topic']}\n\n"
        f"Include an intro, sections per theme, and a conclusion. Keep the "
        f"inline source URLs.\n\nFindings:\n{findings_block}"
    )
    return {"report": report.content}


builder = StateGraph(ResearchState)
builder.add_node("planner", planner)
builder.add_node("researcher", researcher)
builder.add_node("synthesizer", synthesizer)

builder.add_edge(START, "planner")
builder.add_conditional_edges("planner", fan_out, ["researcher"])
builder.add_edge("researcher", "synthesizer")
builder.add_edge("synthesizer", END)

graph = builder.compile()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: uv run main.py "your research topic"')
        sys.exit(1)

    topic = sys.argv[1]
    result = graph.invoke({"topic": topic, "findings": []})

    print("Sub-questions researched:")
    for q in result["sub_questions"]:
        print(f"  - {q}")

    with open("report.md", "w") as f:
        f.write(result["report"])
    print("Report saved to report.md")
