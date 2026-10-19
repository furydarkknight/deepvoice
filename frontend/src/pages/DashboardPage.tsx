import { useEffect, useState } from 'react'
import axios from 'axios'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { Clock, Trash2, CheckCircle, AlertTriangle, XCircle } from 'lucide-react'
import clsx from 'clsx'

interface HistoryEntry {
  id:           string
  timestamp:    string
  filename:     string
  is_fake:      boolean
  confidence:   number
  risk_level:   'SAFE' | 'SUSPICIOUS' | 'HIGH'
  duration_sec: number
}

const RISK_ICON = {
  SAFE:       { icon: CheckCircle,   cls: 'text-safe-400'   },
  SUSPICIOUS: { icon: AlertTriangle, cls: 'text-warn-400'   },
  HIGH:       { icon: XCircle,       cls: 'text-danger-400' },
}

const BAR_COLOUR = {
  SAFE:       '#22c55e',
  SUSPICIOUS: '#f59e0b',
  HIGH:       '#ef4444',
}

function fmt(iso: string) {
  return new Date(iso).toLocaleString(undefined, {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  })
}

export default function DashboardPage() {
  const [history,  setHistory]  = useState<HistoryEntry[]>([])
  const [loading,  setLoading]  = useState(true)

  const load = async () => {
    setLoading(true)
    try {
      const { data } = await axios.get<HistoryEntry[]>('/api/history?limit=100')
      setHistory(data)
    } catch {
      // backend not reachable — show empty state
    } finally {
      setLoading(false)
    }
  }

  const clearAll = async () => {
    await axios.delete('/api/history').catch(() => {})
    setHistory([])
  }

  useEffect(() => { load() }, [])

  // Stats
  const total     = history.length
  const fakes     = history.filter(h => h.is_fake).length
  const reals     = total - fakes
  const avgConf   = total ? (history.reduce((s, h) => s + h.confidence, 0) / total * 100).toFixed(1) : '—'

  // Chart data (last 20)
  const chartData = history.slice(0, 20).reverse().map((h, i) => ({
    name: `#${i + 1}`,
    confidence: Math.round(h.confidence * 100),
    risk: h.risk_level,
  }))

  return (
    <div className="max-w-5xl mx-auto px-4 py-16">
      <div className="flex items-center justify-between mb-10">
        <div>
          <h1 className="text-3xl font-bold text-white">Call Dashboard</h1>
          <p className="text-gray-500 mt-1">History of all analysed recordings</p>
        </div>
        {total > 0 && (
          <button onClick={clearAll} className="btn-ghost text-sm text-danger-400 hover:text-danger-300">
            <Trash2 size={15} /> Clear All
          </button>
        )}
      </div>

      {/* Stats bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-10">
        {[
          { label: 'Total Analysed', value: total, cls: 'text-white' },
          { label: 'Real Voices',    value: reals,  cls: 'text-safe-400'   },
          { label: 'Fake Detected',  value: fakes,  cls: 'text-danger-400' },
          { label: 'Avg Confidence', value: avgConf + '%', cls: 'text-brand-400' },
        ].map(({ label, value, cls }) => (
          <div key={label} className="card text-center">
            <p className={clsx('text-2xl font-black', cls)}>{value}</p>
            <p className="text-gray-500 text-xs mt-1">{label}</p>
          </div>
        ))}
      </div>

      {/* Confidence chart */}
      {chartData.length > 0 && (
        <div className="card mb-8">
          <p className="text-sm font-semibold text-gray-400 mb-4">Fake Confidence — Last 20 Analyses</p>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={chartData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
              <XAxis dataKey="name" tick={{ fill: '#6b7280', fontSize: 11 }} />
              <YAxis domain={[0, 100]} tick={{ fill: '#6b7280', fontSize: 11 }} unit="%" />
              <Tooltip
                contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }}
                labelStyle={{ color: '#e5e7eb' }}
                formatter={(v: number) => [`${v}%`, 'Fake probability']}
              />
              <Bar dataKey="confidence" radius={[4, 4, 0, 0]}>
                {chartData.map((d, i) => (
                  <Cell key={i} fill={BAR_COLOUR[d.risk] || '#3b82f6'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* History table */}
      {loading ? (
        <p className="text-gray-500 text-center py-10">Loading…</p>
      ) : total === 0 ? (
        <div className="card text-center py-16">
          <Clock size={36} className="mx-auto text-gray-700 mb-3" />
          <p className="text-gray-500">No calls analysed yet.</p>
          <p className="text-gray-600 text-sm mt-1">Upload a recording to get started.</p>
        </div>
      ) : (
        <div className="card overflow-hidden p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="border-b border-gray-800">
                <tr className="text-gray-500 text-xs uppercase tracking-wide">
                  <th className="px-5 py-3 text-left">File</th>
                  <th className="px-4 py-3 text-left">Time</th>
                  <th className="px-4 py-3 text-left">Duration</th>
                  <th className="px-4 py-3 text-left">Risk</th>
                  <th className="px-4 py-3 text-right">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {history.map(h => {
                  const { icon: Icon, cls } = RISK_ICON[h.risk_level]
                  return (
                    <tr key={h.id} className="border-b border-gray-800/60 hover:bg-gray-800/30 transition-colors">
                      <td className="px-5 py-3 font-medium text-white truncate max-w-[180px]">
                        {h.filename}
                      </td>
                      <td className="px-4 py-3 text-gray-500 whitespace-nowrap">{fmt(h.timestamp)}</td>
                      <td className="px-4 py-3 text-gray-500">{h.duration_sec}s</td>
                      <td className="px-4 py-3">
                        <span className={clsx('flex items-center gap-1.5 font-semibold', cls)}>
                          <Icon size={14} /> {h.risk_level}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right font-mono font-bold">
                        <span className={cls}>{Math.round(h.confidence * 100)}%</span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
