import { useState } from 'react'
import axios from 'axios'
import AudioDropzone from '@/components/AudioDropzone'
import ResultCard from '@/components/ResultCard'
import { Loader2, UploadCloud } from 'lucide-react'

interface AnalysisResult {
  is_fake:            boolean
  confidence:         number
  risk_level:         'SAFE' | 'SUSPICIOUS' | 'HIGH'
  segment_scores:     number[]
  processing_time_ms: number
  sample_rate:        number
  duration_sec:       number
}

export default function UploadPage() {
  const [file,    setFile]    = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [result,  setResult]  = useState<AnalysisResult | null>(null)
  const [error,   setError]   = useState<string | null>(null)

  const handleFile = (f: File) => {
    setFile(f)
    setResult(null)
    setError(null)
  }

  const analyse = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)

    const form = new FormData()
    form.append('audio', file)

    try {
      const { data } = await axios.post<AnalysisResult>('/api/analyze', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setResult(data)
    } catch (e: any) {
      setError(e.response?.data?.detail || 'Analysis failed. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto px-4 py-16">
      <div className="mb-10 text-center">
        <h1 className="text-3xl font-bold text-white mb-2">Analyse a Recording</h1>
        <p className="text-gray-500">
          Upload a phone call recording to detect AI-generated voices.
        </p>
      </div>

      <AudioDropzone onFile={handleFile} />

      {file && (
        <div className="mt-5 flex justify-center">
          <button
            onClick={analyse}
            disabled={loading}
            className="btn-primary text-base px-8 py-3"
          >
            {loading ? (
              <><Loader2 size={18} className="animate-spin" /> Analysing…</>
            ) : (
              <><UploadCloud size={18} /> Run Detection</>
            )}
          </button>
        </div>
      )}

      {error && (
        <div className="mt-6 p-4 bg-danger-500/10 border border-danger-500/30 rounded-xl text-danger-400 text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="mt-8">
          <ResultCard
            isFake={result.is_fake}
            confidence={result.confidence}
            riskLevel={result.risk_level}
            segmentScores={result.segment_scores}
            durationSec={result.duration_sec}
            processingMs={result.processing_time_ms}
          />
        </div>
      )}

      {/* Tips */}
      <div className="mt-10 card text-sm text-gray-500 space-y-1.5">
        <p className="font-semibold text-gray-300 mb-2">Tips for best results</p>
        <p>• Use a clean recording with minimal background noise.</p>
        <p>• At least 3–4 seconds of speech is needed per analysis.</p>
        <p>• Supported formats: WAV (recommended), MP3, OGG, FLAC.</p>
        <p>• Codec-compressed audio (like WhatsApp voice notes) may reduce accuracy slightly.</p>
      </div>
    </div>
  )
}
