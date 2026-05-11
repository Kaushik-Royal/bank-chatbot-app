# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from doc_load import DocLoad
from agent import build_graph
from langchain_core.messages import HumanMessage

app = Flask(__name__)
CORS(app)  # Enable CORS for communication with React frontend

# Initialise DocLoad (connects to existing ChromaDB) and compile the LangGraph agent
loadDocs = DocLoad()
agent_graph = build_graph(loadDocs)

# In-memory conversation history (resets on server restart)
conversation_history = []


@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'response': 'Please enter a question.'})

    # Append the user message to history before invoking the graph
    conversation_history.append(HumanMessage(content=query))

    result = agent_graph.invoke({
        "messages": list(conversation_history),
        "query":    query,
        "intent":   "",
        "context":  "",
        "answer":   "",
    })

    answer = result["answer"]

    # result["messages"] = conversation_history + [AIMessage(answer)]
    # Sync the AI response back into our history list
    new_count = len(result["messages"]) - len(conversation_history)
    if new_count > 0:
        conversation_history.extend(result["messages"][-new_count:])

    return jsonify({'response': answer})


@app.route('/reset', methods=['POST'])
def reset_conversation():
    """Clear conversation history (called by the frontend 'New Chat' button)."""
    conversation_history.clear()
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    # Documents are already embedded in bank_db1/. To re-load docs, run doc_load.py directly.
    app.run(debug=False)
