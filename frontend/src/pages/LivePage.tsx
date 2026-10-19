import { useEffect, useRef, useState } from 'react'
import { Mic, MicOff, Activity } from 'lucide-react'
import WaveAnimation from '@/components/WaveAnimation'
import clsx from 'clsx'

interface ChunkResult {
  chunk_id:   number
  is_fake:    boolean
  confidence: number
  risk_level: 'SAFE' | 'SUSPICIOUS' | 'HIGH'
}

const SAMPLE_RATE  = 16000
const CHUNK_SEC    = 4
const CHUNK_FRAMES = SAMPLE_RATE * CHUNK_SEC

const RISK_STYLE = {
  SAFE:       'text-safe-400   border-safe-500/30   bg-safe-500/10',
  SUSPICIOUS: 'text-warn-400   border-warn-500/30   bg-warn-500/10',
  HIGH:       'text-danger-400 border-danger-500/30 bg-danger-500/10',
}

export default function LivePage() {
  const [active,    setActive]    = useState(false)
  const [results,   setResults]   = useState<ChunkResult[]>([])
  const [statusMsg, setStatusMsg] = useState('Click Start to begin live detection')
  const [error,     setError]     = useState<string | null>(null)

  const wsRef      = useRef<WebSocket | null>(null)
  const streamRef  = useRef<MediaStream | null>(null)
  const ctxRef     = useRef<AudioContext | null>(null)
  const bufRef     = useRef<Float32Array[]>([])
  const chunkIdRef = useRef(0)

  const stop = () => {
    streamRef.current?.getTracks().forEach(t => t.stop())
    ctxRef.current?.close()
    wsRef.current?.send('close')
    wsRef.current?.close()
    wsRef.current  = null
    streamRef.current = null
    ctxRef.current    = null
    bufRef.current    = []
    setActive(false)
    setStatusMsg('Detection stopped')
  }

  const start = async () => {
    setError(null)
    setResults([])
    chunkIdRef.current = 0

    // Microphone access
    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: SAMPLE_RATE, channelCount: 1 } })
    } catch {
      setError('Microphone access denied. Please allow microphone access in your browser.')
      return
    }
    streamRef.current = stream

    // WebSocket
    const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${wsProtocol}://${window.location.host}/ws/live`)
    wsRef.current = ws

    ws.onopen  = () => setStatusMsg('Connected — listening…')
    ws.onerror = () => setError('WebSocket error — is the backend running?')
    ws.onclose = () => setStatusMsg(s => s === 'Connected — listening…' ? 'Connection closed' : s)
    ws.onmessage = (evt) => {
      const data: ChunkResult = JSON.parse(evt.data)
      setResults(prev => [data, ...prev].slice(0, 50))
    }

    // Audio worklet to collect raw PCM
    const ctx = new AudioContext({ sampleRate: SAMPLE_RATE })
    ctxRef.current = ctx
    const src = ctx.createMediaStreamSource(stream)

    await ctx.audioWorklet.addModule(
      URL.createObjectURL(new Blob([`
        class Collector extends AudioWorkletProcessor {
          constructor() { super(); this._buf = []; }
          process(inputs) {
            const ch = inputs[0][0];
            if (ch) this._buf.push(...ch);
            if (this._buf.length >= ${CHUNK_FRAMES}) {
              const chunk = new Float32Array(this._buf.splice(0, ${CHUNK_FRAMES}));
              this.port.postMessage(chunk);
            }
            return true;
          }
        }
        registerProcessor('collector', Collector);
      `], { type: 'application/javascript' }))
    )

    const worklet = new AudioWorkletNode(ctx, 'collector')
    worklet.port.onmessage = (evt: MessageEvent<Float32Array>) => {
      const float32 = evt.data
      // Convert to 16-bit PCM
      const int16 = new Int16Array(float32.length)
      float32.forEach((v, i) => { int16[i] = Math.max(-32768, Math.min(32767, v * 32768)) })
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(int16.buffer)
      }
    }

    src.connect(worklet)
    setActive(true)
    setStatusMsg('Connected — listening…')
  }

  useEffect(() => () => { stop() }, [])

  const latest = results[0]

  return (
    <div className="max-w-3xl mx-auto px-4 py-16">
      <div className="text-center mb-10">
        <h1 className="text-3xl font-bold text-white mb-2">Live Call Detection</h1>
        <p className="text-gray-500">
          Stream your microphone and get real-time deepfake scoring every 4 seconds.
        </p>
      </div>

      {/* Controls */}
      <div className="card flex flex-col items-center gap-6 mb-8">
        <WaveAnimation active={active} />

        <p className={clsx('text-sm font-medium', active ? 'text-brand-400' : 'text-gray-500')}>
          {statusMsg}
        </p>

        <button
          onClick={active ? stop : start}
          className={clsx(
            'flex items-center gap-2 px-8 py-3 rounded-xl font-bold text-base transition-colors',
            active
              ? 'bg-danger-600 hover:bg-danger-700 text-white'
              : 'btn-primary'
          )}
        >
          {active ? <><MicOff size={18} /> Stop</> : <><Mic size={18} /> Start Detection</>}
        </button>

        {error && (
          <p className="text-danger-400 text-sm bg-danger-500/10 border border-danger-500/30 rounded-xl px-4 py-2">
            {error}
          </p>
        )}
      </div>

      {/* Latest result */}
      {latest && (
        <div className={clsx('card border mb-6', RISK_STYLE[latest.risk_level])}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Activity size={20} />
              <div>
                <p className="font-bold">Latest Chunk #{latest.chunk_id + 1}</p>
                <p className="text-sm opacity-80">
                  {latest.risk_level} · {Math.round(latest.confidence * 100)}% fake probability
                </p>
              </div>
            </div>
            <span className="text-3xl font-black">{Math.round(latest.confidence * 100)}%</span>
          </div>
        </div>
      )}

      {/* History */}
      {results.length > 0 && (
        <div className="card">
          <p className="text-sm font-semibold text-gray-400 mb-3">Chunk History</p>
          <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
            {results.map(r => (
              <div key={r.chunk_id} className="flex items-center justify-between text-sm">
                <span className="text-gray-500">Chunk #{r.chunk_id + 1}</span>
                <div className="flex items-center gap-3">
                  <div className="h-2 w-32 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className={clsx('h-full rounded-full', {
                        'bg-safe-500':   r.risk_level === 'SAFE',
                        'bg-warn-500':   r.risk_level === 'SUSPICIOUS',
                        'bg-danger-500': r.risk_level === 'HIGH',
                      })}
                      style={{ width: `${Math.round(r.confidence * 100)}%` }}
                    />
                  </div>
                  <span className={clsx('font-mono font-bold w-10 text-right', {
                    'text-safe-400':   r.risk_level === 'SAFE',
                    'text-warn-400':   r.risk_level === 'SUSPICIOUS',
                    'text-danger-400': r.risk_level === 'HIGH',
                  })}>
                    {Math.round(r.confidence * 100)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
