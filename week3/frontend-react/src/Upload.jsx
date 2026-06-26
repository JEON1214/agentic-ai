import { useState } from 'react'

export default function Upload({ onUploaded }) {
  const [file, setFile] = useState(null)
  const [status, setStatus] = useState('')

  const upload = async () => {
    if (!file) return setStatus('Select a file')
    const fd = new FormData()
    fd.append('file', file)
    fd.append('collection', 'video_transcripts')
    fd.append('user_id', 'user1')

    setStatus('Uploading...')
    try {
      const res = await fetch('http://127.0.0.1:8000/upload', { method: 'POST', body: fd })
      const js = await res.json()
      if (res.ok) {
        setStatus('Uploaded')
        onUploaded(js.session_id)
      } else {
        setStatus(js.detail || 'Upload failed')
      }
    } catch (e) {
      setStatus('Upload error')
    }
  }

  return (
    <div className="upload-card">
      <input type="file" accept=".txt" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      <button onClick={upload}>Upload file</button>
      <div>{status}</div>
    </div>
  )
}
