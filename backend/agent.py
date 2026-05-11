# backend/agent.py
#
# LangGraph-based agentic pipeline for the bank chatbot.
#
# Graph layout:
#
#   [classify] ──rag──────► [rag_search] ──► [generate] ──► END
#              ──calculator─► [calculator] ──► [generate]
#              ──unknown────────────────────► [generate]
#
# State flows through three nodes:
#   1. classify  – LLM picks intent: "rag" | "calculator" | "unknown"
#   2. tool node – retrieves context (vector search or EMI calc)
#   3. generate  – LLM writes the final answer using context + chat history

import json
import re
from typing import Annotated, List
import operator

import ollama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

MODEL = "deepseek-r1:1.5b"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # Conversation history – each node can append messages (operator.add merges lists)
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    intent: str      # "rag" | "calculator" | "unknown"
    context: str     # retrieved docs or calculation result
    answer: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Strip DeepSeek <think> blocks and extract text after 'Answer:' if present."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"Answer:\s*(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return re.sub(r"\*\*|\*|`", "", text).strip()


def _ollama(prompt: str) -> str:
    resp = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"]


# ---------------------------------------------------------------------------
# Node 1 – Classify intent
# ---------------------------------------------------------------------------

def classify_node(state: AgentState) -> dict:
    query = state["query"]
    prompt = (
        "You are classifying a banking chatbot query. Reply with ONLY one word.\n\n"
        "Categories:\n"
        "- rag        : policy questions, interest rates, fees, account rules, loan terms, document-based info\n"
        "- calculator : compute EMI, monthly payment, total interest, loan comparison\n"
        "- unknown    : greetings, small-talk, or anything not banking-related\n\n"
        f"Query: {query}\n"
        "Category:"
    )
    raw = _clean(_ollama(prompt)).lower()

    if "calculator" in raw:
        intent = "calculator"
    elif "rag" in raw:
        intent = "rag"
    else:
        intent = "unknown"

    print(f"[Agent] intent → {intent}")
    return {"intent": intent}


# ---------------------------------------------------------------------------
# Node 2a – RAG retrieval
# ---------------------------------------------------------------------------

def rag_node(state: AgentState, vector_store) -> dict:
    results = vector_store.similarity_search(state["query"])
    context = "\n\n".join(doc.page_content for doc in results)
    print(f"[Agent] RAG retrieved {len(results)} chunks")
    return {"context": context}


# ---------------------------------------------------------------------------
# Node 2b – EMI / loan calculator
# ---------------------------------------------------------------------------

def calculator_node(state: AgentState) -> dict:
    query = state["query"]
    prompt = (
        "Extract loan parameters from this query. "
        "Return ONLY valid JSON with these keys:\n"
        "  principal           – loan amount as a number (null if not mentioned)\n"
        "  annual_rate_percent – interest rate as a percentage number e.g. 8.5 (null if not mentioned)\n"
        "  tenure_months       – loan duration in months as a number (null if not mentioned)\n\n"
        f"Query: {query}\nJSON:"
    )
    raw = _clean(_ollama(prompt))

    try:
        m = re.search(r"\{[^}]+\}", raw, re.DOTALL)
        if m:
            params = json.loads(m.group())
            p = params.get("principal")
            r_pct = params.get("annual_rate_percent")
            n = params.get("tenure_months")

            if p and r_pct and n:
                p, n = float(p), int(n)
                r = float(r_pct) / 100 / 12          # monthly rate
                emi = (p * r * (1 + r) ** n) / ((1 + r) ** n - 1)
                total = emi * n
                context = (
                    f"EMI Calculation Result:\n"
                    f"  Principal             : ₹{p:,.0f}\n"
                    f"  Annual Interest Rate  : {r_pct}%\n"
                    f"  Tenure                : {n} months "
                    f"({n // 12} yr {n % 12} mo)\n"
                    f"  Monthly EMI           : ₹{emi:,.2f}\n"
                    f"  Total Payment         : ₹{total:,.2f}\n"
                    f"  Total Interest Paid   : ₹{total - p:,.2f}"
                )
                return {"context": context}
    except (json.JSONDecodeError, KeyError, ZeroDivisionError, TypeError):
        pass

    return {
        "context": (
            "Could not extract calculation parameters. "
            "Please provide the principal amount, interest rate (%), and tenure (months or years)."
        )
    }


# ---------------------------------------------------------------------------
# Node 3 – Generate final answer
# ---------------------------------------------------------------------------

def generate_node(state: AgentState) -> dict:
    query   = state["query"]
    context = state.get("context", "")
    intent  = state.get("intent", "unknown")
    history = state.get("messages", [])

    # Build a short conversation history string (last 4 messages, excluding current)
    prior = history[:-1][-4:] if len(history) > 1 else []
    history_str = ""
    if prior:
        history_str = "Previous conversation:\n"
        for msg in prior:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            history_str += f"{role}: {msg.content}\n"
        history_str += "\n"

    if intent == "calculator":
        prompt = (
            f"{history_str}User asked: {query}\n\n"
            f"Calculation result:\n{context}\n\n"
            "Explain this result clearly and helpfully in 2-3 sentences."
        )
    elif context:
        prompt = (
            f"{history_str}"
            "Use the bank document context below to answer the question concisely.\n"
            "If the answer is not in the context, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\nAnswer:"
        )
    else:
        prompt = (
            f"{history_str}"
            f"You are a helpful bank assistant. Answer concisely:\n{query}"
        )

    answer = _clean(_ollama(prompt))
    print("[Agent] answer generated")
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


# ---------------------------------------------------------------------------
# Routing function (classify → tool node)
# ---------------------------------------------------------------------------

def _route(state: AgentState) -> str:
    intent = state.get("intent", "unknown")
    if intent == "calculator":
        return "calculator"
    if intent == "rag":
        return "rag"
    return "generate"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_graph(doc_loader):
    """
    Build and compile the LangGraph StateGraph.
    doc_loader is an instance of DocLoad; its vector_store is passed to rag_node.
    """
    graph = StateGraph(AgentState)

    graph.add_node("classify",    classify_node)
    graph.add_node("rag",         lambda s: rag_node(s, doc_loader.vector_store))
    graph.add_node("calculator",  calculator_node)
    graph.add_node("generate",    generate_node)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify",
        _route,
        {"rag": "rag", "calculator": "calculator", "generate": "generate"},
    )

    graph.add_edge("rag",        "generate")
    graph.add_edge("calculator", "generate")
    graph.add_edge("generate",   END)

    return graph.compile()
