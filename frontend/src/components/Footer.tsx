import { Github } from 'lucide-react'
import { Link } from 'react-router-dom'

export default function Footer() {
  return (
    <footer className="border-t border-gray-800 py-8 mt-16">
      <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-gray-500">
        <p>© {new Date().getFullYear()} VoiceShield. Open-source deepfake audio detection.</p>
        <div className="flex items-center gap-4">
          <Link to="/about" className="hover:text-gray-300 transition-colors">About</Link>
          <a
            href="https://github.com/YOUR_USERNAME/deepfake-voice-detector"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 hover:text-gray-300 transition-colors"
          >
            <Github size={15} />
            GitHub
          </a>
        </div>
      </div>
    </footer>
  )
}
