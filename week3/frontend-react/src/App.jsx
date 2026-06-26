import { useState } from 'react'
import './App.css'
import Upload from './Upload'
import Chat from './Chat'

function App() {
  const [sessionId, setSessionId] = useState('')

  return (
    <div className="app-root">
      <header>
        <h1>Vector Browser — RAG Chat</h1>
        <p>Upload your transcript, then ask questions and get answers from your own data.</p>
      </header>

      <main style={{ display: 'flex', gap: 24 }}>
        <aside style={{ width: 320 }}>
          <Upload onUploaded={(sid) => setSessionId(sid)} />
          <div style={{ marginTop: 12 }}>Session: {sessionId}</div>
        </aside>
        <section style={{ flex: 1 }}>
          <Chat sessionId={sessionId} />
        </section>
      </main>
    </div>
  )
}

export default App
