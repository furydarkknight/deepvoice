import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { UploadCloud, FileAudio, X } from 'lucide-react'
import clsx from 'clsx'

interface Props {
  onFile: (file: File) => void
}

const ACCEPT = {
  'audio/wav':  ['.wav'],
  'audio/mpeg': ['.mp3'],
  'audio/ogg':  ['.ogg'],
  'audio/flac': ['.flac'],
  'audio/webm': ['.webm'],
}

export default function AudioDropzone({ onFile }: Props) {
  const [preview, setPreview] = useState<File | null>(null)

  const onDrop = useCallback((accepted: File[]) => {
    if (accepted.length) {
      setPreview(accepted[0])
      onFile(accepted[0])
    }
  }, [onFile])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPT,
    maxFiles: 1,
    maxSize: 60 * 1024 * 1024,
  })

  const clear = (e: React.MouseEvent) => {
    e.stopPropagation()
    setPreview(null)
  }

  return (
    <div
      {...getRootProps()}
      className={clsx(
        'relative border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-colors',
        isDragActive
          ? 'border-brand-500 bg-brand-600/10'
          : 'border-gray-700 hover:border-brand-500/60 bg-gray-900/50'
      )}
    >
      <input {...getInputProps()} />

      {preview ? (
        <div className="flex items-center justify-center gap-3">
          <FileAudio size={36} className="text-brand-400" />
          <div className="text-left">
            <p className="font-semibold text-white truncate max-w-xs">{preview.name}</p>
            <p className="text-sm text-gray-500">{(preview.size / 1024).toFixed(1)} KB</p>
          </div>
          <button onClick={clear} className="ml-4 p-1 hover:bg-gray-700 rounded-lg transition-colors">
            <X size={16} className="text-gray-400" />
          </button>
        </div>
      ) : (
        <>
          <UploadCloud size={44} className="mx-auto mb-3 text-gray-600" />
          <p className="text-gray-300 font-medium">
            {isDragActive ? 'Drop audio file here…' : 'Drag & drop or click to upload'}
          </p>
          <p className="text-gray-600 text-sm mt-1">WAV · MP3 · OGG · FLAC · WebM · up to 60 MB</p>
        </>
      )}
    </div>
  )
}
