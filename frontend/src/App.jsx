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
  const [stats, setStats] = useState(null)

  useEffect(() => {
    fetchSessions()
    fetchStats()
  }, [sessionSearch, sessionGoalFilter])

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/stats`)
      if (response.ok) {
        const data = await response.json()
        setStats(data)
      }
    } catch (err) {
      console.error('Error fetching stats:', err)
    }
  }

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

  const deleteSession = async (sessionId) => {
    if (!window.confirm('Are you sure you want to delete this session?')) {
      return
    }
    try {
      const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}`, {
        method: 'DELETE'
      })
      if (response.ok) {
        fetchSessions()
        if (result && result.session_id === sessionId) {
          setResult(null)
          setActiveTab('sessions')
        }
      } else {
        setError('Failed to delete session')
      }
    } catch (err) {
      setError('Failed to delete session')
    }
  }

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text).then(() => {
      alert('Copied to clipboard!')
    }).catch(() => {
      setError('Failed to copy to clipboard')
    })
  }

  const formatRelativeTime = (dateString) => {
    const date = new Date(dateString)
    const now = new Date()
    const diffMs = now - date
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMs / 3600000)
    const diffDays = Math.floor(diffMs / 86400000)

    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`
    if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`
    if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`
    return date.toLocaleDateString()
  }

  const clearForm = (formType) => {
    if (formType === 'video') {
      setSessionName('')
      setError(null)
      const fileInput = document.querySelector('input[type="file"]')
      if (fileInput) fileInput.value = ''
    } else if (formType === 'text') {
      setTextTranscript('')
      setSessionName('')
      setError(null)
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
        <button 
          className={activeTab === 'stats' ? 'active' : ''}
          onClick={() => {
            setActiveTab('stats')
            fetchStats()
          }}
        >
          Stats
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

              <div className="form-actions">
                <button type="submit" disabled={loading}>
                  {loading ? (
                    <>
                      <span className="spinner"></span> Analyzing...
                    </>
                  ) : (
                    'Analyze Video'
                  )}
                </button>
                <button type="button" onClick={() => clearForm('video')} className="clear-btn">
                  Clear
                </button>
              </div>
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
                <label>
                  Transcript: 
                  <span className="char-count">
                    {textTranscript.length} characters
                  </span>
                </label>
                <textarea
                  value={textTranscript}
                  onChange={(e) => setTextTranscript(e.target.value)}
                  placeholder="Enter your speech transcript here..."
                  rows="8"
                  required
                />
              </div>

              <div className="form-actions">
                <button type="submit" disabled={loading}>
                  {loading ? (
                    <>
                      <span className="spinner"></span> Processing...
                    </>
                  ) : (
                    'Get Feedback'
                  )}
                </button>
                <button type="button" onClick={() => clearForm('text')} className="clear-btn">
                  Clear
                </button>
              </div>
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
                    <p><strong>Status:</strong> <span className={`status-badge status-${session.status}`}>{session.status}</span></p>
                    <p><strong>Created:</strong> {formatRelativeTime(session.created_at)}</p>
                    <div className="session-actions">
                      {session.has_results && (
                        <button onClick={() => loadSession(session.session_id)}>
                          View Results
                        </button>
                      )}
                      <button onClick={() => deleteSession(session.session_id)} className="delete-btn">
                        Delete
                      </button>
                    </div>
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
              <div className="result-actions">
                <button onClick={() => copyToClipboard(JSON.stringify(result, null, 2))} className="copy-btn">
                  📋 Copy JSON
                </button>
                <button onClick={exportResult} className="export-btn">
                  📥 Export JSON
                </button>
              </div>
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

        {activeTab === 'stats' && (
          <div className="tab-content">
            <h2>Statistics</h2>
            {stats ? (
              <div className="stats-grid">
                <div className="stat-card">
                  <h3>Total Sessions</h3>
                  <p className="stat-value">{stats.total_sessions}</p>
                </div>
                <div className="stat-card">
                  <h3>Completed</h3>
                  <p className="stat-value">{stats.completed_sessions}</p>
                </div>
                <div className="stat-card">
                  <h3>Processing</h3>
                  <p className="stat-value">{stats.processing_sessions}</p>
                </div>
                <div className="stat-card">
                  <h3>Failed</h3>
                  <p className="stat-value">{stats.failed_sessions}</p>
                </div>
                {stats.average_score !== null && (
                  <div className="stat-card">
                    <h3>Average Score</h3>
                    <p className="stat-value">{stats.average_score}/100</p>
                  </div>
                )}
                {stats.average_processing_time_seconds !== null && (
                  <div className="stat-card">
                    <h3>Avg Processing Time</h3>
                    <p className="stat-value">{stats.average_processing_time_seconds}s</p>
                  </div>
                )}
              </div>
            ) : (
              <p>Loading statistics...</p>
            )}
            {stats && stats.goal_distribution && (
              <div className="result-section">
                <h3>Goal Distribution</h3>
                <div className="goal-distribution">
                  {Object.entries(stats.goal_distribution).map(([goal, count]) => (
                    <div key={goal} className="goal-item">
                      <span className="goal-name">{goal.replace('-', ' ')}</span>
                      <span className="goal-count">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}

export default App
