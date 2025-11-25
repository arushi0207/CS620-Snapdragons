import { useState, useEffect } from 'react'
import './App.css'

const API_BASE_URL = 'http://localhost:8000/api'

function App() {
  const [activeTab, setActiveTab] = useState('upload')
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [selectedGoal, setSelectedGoal] = useState('job-interview')
  const [sessionName, setSessionName] = useState('')
  const [textTranscript, setTextTranscript] = useState('')
  const [result, setResult] = useState(null)
  const [sessionSearch, setSessionSearch] = useState('')
  const [sessionGoalFilter, setSessionGoalFilter] = useState('')

  useEffect(() => {
    fetchSessions()
  }, [sessionSearch, sessionGoalFilter])

  const fetchSessions = async () => {
    try {
      let url = `${API_BASE_URL}/sessions?limit=10`
      if (sessionSearch) {
        url += `&session_name=${encodeURIComponent(sessionSearch)}`
      }
      if (sessionGoalFilter) {
        url += `&goal=${encodeURIComponent(sessionGoalFilter)}`
      }
      const response = await fetch(url)
      if (response.ok) {
        const data = await response.json()
        setSessions(data)
      }
    } catch (err) {
      console.error('Error fetching sessions:', err)
    }
  }

  const handleVideoUpload = async (e) => {
    e.preventDefault()
    const fileInput = e.target.querySelector('input[type="file"]')
    const file = fileInput?.files[0]
    
    if (!file) {
      setError('Please select a video file')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('goal', selectedGoal)
      if (sessionName) {
        formData.append('session_name', sessionName)
      }

      const response = await fetch(`${API_BASE_URL}/analyze_video?goal=${selectedGoal}&session_name=${sessionName || ''}`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to analyze video')
      }

      const data = await response.json()
      setResult(data)
      fetchSessions()
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleTextFeedback = async (e) => {
    e.preventDefault()
    
    if (!textTranscript.trim() || textTranscript.trim().length < 10) {
      setError('Please enter at least 10 characters of transcript')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch(`${API_BASE_URL}/get_text_feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          transcript: textTranscript,
          goal: selectedGoal,
          session_name: sessionName || undefined
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to get feedback')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const exportResult = () => {
    if (!result) return
    
    const dataStr = JSON.stringify(result, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = `speakeasy-analysis-${Date.now()}.json`
    link.click()
    URL.revokeObjectURL(url)
  }

  const loadSession = async (sessionId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}`)
      if (response.ok) {
        const data = await response.json()
        setResult(data)
        setActiveTab('result')
      }
    } catch (err) {
      setError('Failed to load session')
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🎤 SpeakEasy</h1>
        <p>AI-Powered Public Speaking Coach</p>
      </header>

      <nav className="tabs">
        <button 
          className={activeTab === 'upload' ? 'active' : ''}
          onClick={() => setActiveTab('upload')}
        >
          Video Analysis
        </button>
        <button 
          className={activeTab === 'text' ? 'active' : ''}
          onClick={() => setActiveTab('text')}
        >
          Text Feedback
        </button>
        <button 
          className={activeTab === 'sessions' ? 'active' : ''}
          onClick={() => setActiveTab('sessions')}
        >
          Sessions ({sessions.length})
        </button>
        {result && (
          <button 
            className={activeTab === 'result' ? 'active' : ''}
            onClick={() => setActiveTab('result')}
          >
            Results
          </button>
        )}
      </nav>

      <main className="main-content">
        {error && (
          <div className="error-message">
            ⚠️ {error}
          </div>
        )}

        {activeTab === 'upload' && (
          <div className="tab-content">
            <h2>Upload Video for Analysis</h2>
            <form onSubmit={handleVideoUpload}>
              <div className="form-group">
                <label>Goal Type:</label>
                <select 
                  value={selectedGoal} 
                  onChange={(e) => setSelectedGoal(e.target.value)}
                >
                  <option value="job-interview">Job Interview</option>
                  <option value="class-presentation">Class Presentation</option>
                  <option value="networking-pitch">Networking Pitch</option>
                  <option value="general-confidence">General Confidence</option>
                </select>
              </div>

              <div className="form-group">
                <label>Session Name (optional):</label>
                <input
                  type="text"
                  value={sessionName}
                  onChange={(e) => setSessionName(e.target.value)}
                  placeholder="e.g., Practice Session 1"
                />
              </div>

              <div className="form-group">
                <label>Video File:</label>
                <input type="file" accept="video/*" required />
              </div>

              <button type="submit" disabled={loading}>
                {loading ? 'Analyzing...' : 'Analyze Video'}
              </button>
            </form>
          </div>
        )}

        {activeTab === 'text' && (
          <div className="tab-content">
            <h2>Get Feedback from Text</h2>
            <form onSubmit={handleTextFeedback}>
              <div className="form-group">
                <label>Goal Type:</label>
                <select 
                  value={selectedGoal} 
                  onChange={(e) => setSelectedGoal(e.target.value)}
                >
                  <option value="job-interview">Job Interview</option>
                  <option value="class-presentation">Class Presentation</option>
                  <option value="networking-pitch">Networking Pitch</option>
                  <option value="general-confidence">General Confidence</option>
                </select>
              </div>

              <div className="form-group">
                <label>Session Name (optional):</label>
                <input
                  type="text"
                  value={sessionName}
                  onChange={(e) => setSessionName(e.target.value)}
                  placeholder="e.g., Practice Session 1"
                />
              </div>

              <div className="form-group">
                <label>Transcript:</label>
                <textarea
                  value={textTranscript}
                  onChange={(e) => setTextTranscript(e.target.value)}
                  placeholder="Enter your speech transcript here..."
                  rows="8"
                  required
                />
              </div>

              <button type="submit" disabled={loading}>
                {loading ? 'Processing...' : 'Get Feedback'}
              </button>
            </form>
          </div>
        )}

        {activeTab === 'sessions' && (
          <div className="tab-content">
            <h2>Recent Sessions</h2>
            
            <div className="session-filters">
              <div className="form-group">
                <label>Search by Name:</label>
                <input
                  type="text"
                  value={sessionSearch}
                  onChange={(e) => setSessionSearch(e.target.value)}
                  placeholder="Search sessions..."
                />
              </div>
              <div className="form-group">
                <label>Filter by Goal:</label>
                <select 
                  value={sessionGoalFilter} 
                  onChange={(e) => setSessionGoalFilter(e.target.value)}
                >
                  <option value="">All Goals</option>
                  <option value="job-interview">Job Interview</option>
                  <option value="class-presentation">Class Presentation</option>
                  <option value="networking-pitch">Networking Pitch</option>
                  <option value="general-confidence">General Confidence</option>
                </select>
              </div>
            </div>

            {sessions.length === 0 ? (
              <p>No sessions found. {sessionSearch || sessionGoalFilter ? 'Try adjusting your filters.' : 'Upload a video or submit text feedback to get started!'}</p>
            ) : (
              <div className="sessions-list">
                {sessions.map((session) => (
                  <div key={session.session_id} className="session-card">
                    <h3>{session.session_name || 'Unnamed Session'}</h3>
                    <p><strong>Goal:</strong> {session.goal.replace('-', ' ')}</p>
                    <p><strong>Status:</strong> {session.status}</p>
                    <p><strong>Created:</strong> {new Date(session.created_at).toLocaleString()}</p>
                    {session.has_results && (
                      <button onClick={() => loadSession(session.session_id)}>
                        View Results
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === 'result' && result && (
          <div className="tab-content">
            <div className="result-header">
              <h2>Analysis Results</h2>
              <button onClick={exportResult} className="export-btn">
                📥 Export JSON
              </button>
            </div>

            {result.overall_score !== undefined && (
              <div className="score-display">
                <h3>Overall Score: {result.overall_score}/100</h3>
              </div>
            )}

            {result.summary && (
              <div className="result-section">
                <h3>Summary</h3>
                <p>{result.summary}</p>
              </div>
            )}

            {result.metrics && (
              <div className="result-section">
                <h3>Metrics</h3>
                <div className="metrics-grid">
                  {result.metrics.map((metric, idx) => (
                    <div key={idx} className={`metric-card ${metric.status}`}>
                      <h4>{metric.metric_name}</h4>
                      <p className="metric-value">{metric.value} {metric.unit || ''}</p>
                      <p className="metric-message">{metric.message}</p>
                      {metric.tip && (
                        <p className="metric-tip">💡 {metric.tip}</p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {result.recommendations && result.recommendations.length > 0 && (
              <div className="result-section">
                <h3>Recommendations</h3>
                <ul>
                  {result.recommendations.map((rec, idx) => (
                    <li key={idx}>{rec}</li>
                  ))}
                </ul>
              </div>
            )}

            {result.feedback && (
              <div className="result-section">
                <h3>Feedback</h3>
                <pre>{JSON.stringify(result.feedback, null, 2)}</pre>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}

export default App
