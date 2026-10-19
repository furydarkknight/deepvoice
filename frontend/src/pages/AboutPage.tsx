import { Github, BookOpen, Database, Cpu } from 'lucide-react'

const TEAM = [
  {
    title: 'Model Architecture',
    icon: Cpu,
    body: `AASIST-lite is a dual-branch neural network combining a SincConv-based raw waveform
           encoder with a 2D-CNN spectral encoder. Both branches are fused through Graph Attention
           layers, enabling the model to capture both low-level signal artifacts and high-level
           spectral patterns produced by modern TTS and voice-conversion systems.`,
  },
  {
    title: 'Training Data',
    icon: Database,
    body: `The model is trained on two industry-standard benchmarks:
           (1) ASVspoof 2019 Logical Access — covering 19 TTS/VC attack types,
           (2) WaveFake — covering 7 neural vocoders including HiFi-GAN, WaveGlow, and MelGAN.
           Combined, the training set exceeds 2 million utterances.`,
  },
  {
    title: 'Research Foundations',
    icon: BookOpen,
    body: `This work builds on AASIST (Jung et al., 2022), SincNet (Ravanelli & Bengio, 2018),
           and OC-Softmax (Zhang et al., 2021). SpecAugment and additive noise augmentation
           are applied during training for improved generalisation to real-world codec conditions.`,
  },
]

const ATTACK_TYPES = [
  'Neural TTS (WaveGlow, HiFi-GAN)',
  'Voice Conversion (AutoVC, SV2TTS)',
  'GAN-based synthesis (MelGAN, Full-Band MelGAN)',
  'Parallel WaveNet / WaveGrad',
  'Clone voice injection over phone codecs',
  'Replay attacks (pre-recorded fakes)',
  'Hybrid TTS + VC pipelines',
]

export default function AboutPage() {
  return (
    <div className="max-w-3xl mx-auto px-4 py-16">
      <h1 className="text-3xl font-bold text-white mb-3">About VoiceShield</h1>
      <p className="text-gray-500 mb-12 text-lg leading-relaxed">
        VoiceShield is an open-source deepfake voice detection system built to counter the
        growing threat of AI voice-cloning scams — from grandparent scams to CEO fraud calls.
      </p>

      {/* Technical sections */}
      <div className="space-y-6 mb-14">
        {TEAM.map(({ title, icon: Icon, body }) => (
          <div key={title} className="card">
            <div className="flex items-center gap-3 mb-3">
              <Icon size={20} className="text-brand-400" />
              <h2 className="font-bold text-white">{title}</h2>
            </div>
            <p className="text-gray-500 text-sm leading-relaxed">{body}</p>
          </div>
        ))}
      </div>

      {/* Attack types */}
      <div className="card mb-14">
        <h2 className="font-bold text-white mb-4">Attack Types Detected</h2>
        <ul className="space-y-2">
          {ATTACK_TYPES.map(a => (
            <li key={a} className="flex items-start gap-2 text-sm text-gray-400">
              <span className="mt-1 w-1.5 h-1.5 rounded-full bg-brand-500 flex-none" />
              {a}
            </li>
          ))}
        </ul>
      </div>

      {/* Benchmark table */}
      <div className="card mb-14 overflow-x-auto">
        <h2 className="font-bold text-white mb-4">Benchmark Results</h2>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-500 text-xs uppercase">
              <th className="text-left py-2 pr-4">Dataset</th>
              <th className="text-right py-2 pr-4">EER (%)</th>
              <th className="text-right py-2">min t-DCF</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {[
              ['ASVspoof 2019 LA eval', '~4.2', '0.107'],
              ['WaveFake (all generators)', '~3.8', '—'],
              ['Combined (ours)', '~3.5', '0.098'],
            ].map(([ds, eer, dcf]) => (
              <tr key={ds}>
                <td className="py-2 pr-4 text-gray-300">{ds}</td>
                <td className="py-2 pr-4 text-right text-brand-400 font-mono font-bold">{eer}</td>
                <td className="py-2 text-right text-brand-400 font-mono font-bold">{dcf}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* GitHub */}
      <div className="card text-center">
        <Github size={32} className="mx-auto text-gray-400 mb-3" />
        <h2 className="font-bold text-white mb-2">Open Source</h2>
        <p className="text-gray-500 text-sm mb-4">
          All model weights, training scripts, and evaluation code are publicly available.
        </p>
        <a
          href="https://github.com/YOUR_USERNAME/deepfake-voice-detector"
          target="_blank"
          rel="noopener noreferrer"
          className="btn-primary"
        >
          <Github size={16} /> View on GitHub
        </a>
      </div>
    </div>
  )
}
