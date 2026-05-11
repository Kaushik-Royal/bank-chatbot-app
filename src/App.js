import React, { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const chatBoxRef = useRef(null); // Ref for auto-scroll

  const handleReset = async () => {
    await fetch('http://localhost:5000/reset', { method: 'POST' });
    setMessages([]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    const newMessages = [...messages, { sender: 'user', text: query }];
    setMessages(newMessages);
    setQuery('');
    setLoading(true);

    try {
      const res = await fetch('http://localhost:5000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });

      const data = await res.json();
      setMessages([...newMessages, { sender: 'bot', text: data.response }]);
    } catch (error) {
      setMessages([...newMessages, { sender: 'bot', text: 'Error fetching response.' }]);
    } finally {
      setLoading(false);
    }
  };

  // Scroll to bottom when messages update
  useEffect(() => {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }
  }, [messages, loading]);

  return (
    <div className="chat-wrapper">
      <div className="chat-header">
        <h1>Bankbot</h1>
        <button className="reset-btn" onClick={handleReset}>New Chat</button>
      </div>
      <div className="chat-container">
        <div className="chat-box" ref={chatBoxRef}>
          {messages.map((msg, index) => (
            <div key={index} className={`chat-message ${msg.sender}`}>
              {msg.text}
            </div>
          ))}
          {loading && <div className="chat-message bot">...</div>}
        </div>
        <form className="chat-input" onSubmit={handleSubmit}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Type your message..."
          />
          <button type="submit">Send</button>
        </form>
      </div>
    </div>
  );
}

export default App;
