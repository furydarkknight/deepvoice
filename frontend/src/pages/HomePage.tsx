import { Link } from 'react-router-dom'
import { Shield, UploadCloud, Mic, Zap, Lock, BarChart2, AlertTriangle } from 'lucide-react'
import { motion } from 'framer-motion'

const FEATURES = [
  {
    icon: Zap,
    title: 'Real-Time Analysis',
    desc: 'Analyse live phone calls in under 50 ms per chunk — before the conversation ends.',
  },
  {
    icon: Shield,
    title: 'Industry-Grade Model',
    desc: 'AASIST-lite architecture trained on ASVspoof 2019 LA + WaveFake. EER < 5%.',
  },
  {
    icon: BarChart2,
    title: 'Explainable Results',
    desc: 'Per-segment heatmaps show exactly which parts of the call triggered detection.',
  },
  {
    icon: Lock,
    title: 'Privacy First',
    desc: 'Audio is processed locally or on your own server. Nothing leaves your environment.',
  },
  {
    icon: AlertTriangle,
    title: 'Scam Alerts',
    desc: 'Instant risk scoring — SAFE / SUSPICIOUS / HIGH — with confidence percentage.',
  },
  {
    icon: Mic,
    title: 'Live Microphone Mode',
    desc: 'Stream your microphone directly to the detector for real-time call protection.',
  },
]

const STATS = [
  { value: '<5%',    label: 'Equal Error Rate' },
  { value: '~42ms',  label: 'Inference latency' },
  { value: '2M+',    label: 'Training samples' },
  { value: '7+',     label: 'Attack types detected' },
]

export default function HomePage() {
  return (
    <div>
      {/* ── Hero ── */}
      <section className="relative overflow-hidden bg-gradient-to-b from-gray-950 to-gray-900 py-28 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <span className="inline-flex items-center gap-2 bg-brand-600/20 text-brand-400 text-sm font-semibold px-4 py-1.5 rounded-full border border-brand-600/30 mb-6">
              <Shield size={14} /> AI-Powered Voice Authentication
            </span>

            <h1 className="text-4xl sm:text-6xl font-black text-white mb-6 leading-tight">
              Stop{' '}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-brand-400 to-purple-400">
                Deepfake Voice
              </span>{' '}
              Scams Before They Fool You
            </h1>

            <p className="text-gray-400 text-lg sm:text-xl max-w-2xl mx-auto mb-10">
              VoiceShield uses state-of-the-art neural networks to detect AI-cloned and
              synthesised voices in phone calls — protecting individuals and businesses
              from billion-dollar voice fraud.
            </p>

            <div className="flex flex-wrap justify-center gap-4">
              <Link to="/upload"    className="btn-primary text-base px-7 py-3">
                <UploadCloud size={18} /> Analyse a Recording
              </Link>
              <Link to="/live" className="btn-ghost  text-base px-7 py-3 border border-gray-700">
                <Mic size={18} /> Live Detection
              </Link>
            </div>
          </motion.div>
        </div>

        {/* Decorative grid */}
        <div
          className="absolute inset-0 -z-10 opacity-5"
          style={{
            backgroundImage: 'linear-gradient(#3b82f6 1px,transparent 1px),linear-gradient(90deg,#3b82f6 1px,transparent 1px)',
            backgroundSize: '40px 40px',
          }}
        />
      </section>

      {/* ── Stats ── */}
      <section className="bg-gray-900 border-y border-gray-800 py-10 px-4">
        <div className="max-w-5xl mx-auto grid grid-cols-2 sm:grid-cols-4 gap-8 text-center">
          {STATS.map(({ value, label }) => (
            <div key={label}>
              <p className="text-3xl font-black text-brand-400">{value}</p>
              <p className="text-gray-500 text-sm mt-1">{label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Features ── */}
      <section className="py-20 px-4 max-w-7xl mx-auto">
        <div className="text-center mb-14">
          <h2 className="text-3xl font-bold text-white mb-3">Everything You Need</h2>
          <p className="text-gray-500 max-w-xl mx-auto">
            From upload to live streaming — complete protection against voice-cloning fraud.
          </p>
        </div>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {FEATURES.map(({ icon: Icon, title, desc }) => (
            <motion.div
              key={title}
              whileHover={{ y: -4 }}
              className="card group hover:border-brand-600/40 transition-colors"
            >
              <div className="p-2.5 bg-brand-600/15 rounded-xl w-fit mb-4 group-hover:bg-brand-600/25 transition-colors">
                <Icon size={22} className="text-brand-400" />
              </div>
              <h3 className="font-bold text-white mb-2">{title}</h3>
              <p className="text-gray-500 text-sm leading-relaxed">{desc}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* ── How it works ── */}
      <section className="bg-gray-900/60 border-y border-gray-800 py-20 px-4">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold text-white text-center mb-14">How It Works</h2>
          <div className="relative">
            {/* Timeline line */}
            <div className="absolute left-5 top-0 bottom-0 w-px bg-gray-700 hidden sm:block" />
            {[
              {
                step: '01',
                title: 'Audio Ingested',
                body: 'Upload a recording or stream live mic input. We accept WAV, MP3, OGG, and FLAC.',
              },
              {
                step: '02',
                title: 'Feature Extraction',
                body: 'The audio is split into 4-second overlapping chunks. LFCC and raw waveform features are computed for each chunk.',
              },
              {
                step: '03',
                title: 'Neural Network Analysis',
                body: 'AASIST-lite processes both the sinc-filtered waveform and LFCC spectrogram through graph attention layers.',
              },
              {
                step: '04',
                title: 'Risk Verdict',
                body: 'Each chunk receives a fake-probability score. The 75th-percentile score drives the overall SAFE / SUSPICIOUS / HIGH verdict.',
              },
            ].map(({ step, title, body }) => (
              <div key={step} className="flex gap-6 mb-10 relative">
                <div className="flex-none w-10 h-10 rounded-full bg-brand-600 flex items-center justify-center text-white font-bold text-sm z-10">
                  {step}
                </div>
                <div className="pb-2">
                  <h4 className="font-bold text-white mb-1">{title}</h4>
                  <p className="text-gray-500 text-sm leading-relaxed">{body}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="py-20 px-4 text-center">
        <h2 className="text-3xl font-bold text-white mb-4">
          Ready to Protect Your Calls?
        </h2>
        <p className="text-gray-500 mb-8 max-w-md mx-auto">
          Start detecting deepfake voices in seconds — no sign-up required.
        </p>
        <Link to="/upload" className="btn-primary text-lg px-10 py-4">
          <UploadCloud size={20} /> Get Started Free
        </Link>
      </section>
    </div>
  )
}
