import clsx from 'clsx'
import { CheckCircle, AlertTriangle, XCircle } from 'lucide-react'

interface Props {
  isFake:     boolean
  confidence: number
  riskLevel:  'SAFE' | 'SUSPICIOUS' | 'HIGH'
  segmentScores: number[]
  durationSec: number
  processingMs: number
}

const RISK_CONFIG = {
  SAFE:       { label: 'SAFE — Real Voice',      icon: CheckCircle,    bar: 'bg-safe-500',   text: 'text-safe-400',   badge: 'badge-safe' },
  SUSPICIOUS: { label: 'SUSPICIOUS',             icon: AlertTriangle,  bar: 'bg-warn-500',   text: 'text-warn-400',   badge: 'badge-suspicious' },
  HIGH:       { label: 'HIGH RISK — Fake Voice', icon: XCircle,        bar: 'bg-danger-500', text: 'text-danger-400', badge: 'badge-high' },
}

export default function ResultCard({ isFake, confidence, riskLevel, segmentScores, durationSec, processingMs }: Props) {
  const cfg = RISK_CONFIG[riskLevel]
  const Icon = cfg.icon
  const pct  = Math.round(confidence * 100)

  return (
    <div className={clsx('card border-2', {
      'border-safe-500/40':   riskLevel === 'SAFE',
      'border-warn-500/40':   riskLevel === 'SUSPICIOUS',
      'border-danger-500/40': riskLevel === 'HIGH',
    })}>
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <Icon size={28} className={cfg.text} />
        <div>
          <h3 className={clsx('text-lg font-bold', cfg.text)}>{cfg.label}</h3>
          <p className="text-gray-500 text-sm">{durationSec}s analysed · {processingMs}ms inference</p>
        </div>
        <span className={clsx('ml-auto text-2xl font-black', cfg.text)}>{pct}%</span>
      </div>

      {/* Confidence bar */}
      <div className="mb-5">
        <div className="flex justify-between text-xs text-gray-500 mb-1">
          <span>Fake probability</span>
          <span>{pct}%</span>
        </div>
        <div className="h-3 bg-gray-800 rounded-full overflow-hidden">
          <div
            className={clsx('h-full rounded-full transition-all duration-700', cfg.bar)}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {/* Segment heatmap */}
      <div>
        <p className="text-xs text-gray-500 mb-2">Segment analysis ({segmentScores.length} chunks)</p>
        <div className="flex gap-1 flex-wrap">
          {segmentScores.map((s, i) => {
            const p = Math.round(s * 100)
            const bg =
              s < 0.4  ? 'bg-safe-500/70'
              : s < 0.7 ? 'bg-warn-500/70'
              : 'bg-danger-500/70'
            return (
              <div
                key={i}
                title={`Chunk ${i + 1}: ${p}% fake`}
                className={clsx('h-6 rounded cursor-default transition-all', bg)}
                style={{ width: `${Math.max(18, 400 / segmentScores.length)}px` }}
              />
            )
          })}
        </div>
        <p className="text-xs text-gray-600 mt-1.5">
          <span className="inline-block w-3 h-3 bg-safe-500/70 rounded mr-1" />Real
          <span className="inline-block w-3 h-3 bg-warn-500/70 rounded mx-1 ml-3" />Suspicious
          <span className="inline-block w-3 h-3 bg-danger-500/70 rounded mx-1 ml-3" />Fake
        </p>
      </div>
    </div>
  )
}
