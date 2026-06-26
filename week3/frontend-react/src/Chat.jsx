import { useState } from 'react'

export default function Chat({ sessionId }) {
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState([])

  const send = async () => {
    if (!query) return
    setMessages((m) => [...m, { who: 'you', text: query }])
    try {
      const res = await fetch('http://127.0.0.1:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, collection: 'video_transcripts', user_id: 'user1', session_id: sessionId, limit: 3 }),
      })
      const js = await res.json()
      setMessages((m) => [...m, { who: 'agent', text: js.answer }])
      setQuery('')
    } catch (e) {
      setMessages((m) => [...m, { who: 'agent', text: 'Error contacting backend' }])
    }
  }

  return (
    <div className="chat-card">
      <div style={{ minHeight: 120, border: '1px solid #ddd', padding: 8 }}>
        {messages.map((m, i) => (
          <div key={i}><b>{m.who}:</b> {m.text}</div>
        ))}
      </div>
      <input value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && send()} />
      <button onClick={send}>Send</button>
    </div>
  )
}
