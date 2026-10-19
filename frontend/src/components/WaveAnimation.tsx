import { motion } from 'framer-motion'

interface Props { active: boolean }

export default function WaveAnimation({ active }: Props) {
  const bars = Array.from({ length: 28 })
  return (
    <div className="flex items-center justify-center gap-0.5 h-12">
      {bars.map((_, i) => (
        <motion.div
          key={i}
          className="w-1 rounded-full bg-brand-500"
          animate={active ? {
            scaleY: [0.3, 1.2, 0.4, 1.0, 0.3],
            opacity: [0.6, 1, 0.7, 1, 0.6],
          } : { scaleY: 0.3, opacity: 0.3 }}
          transition={{
            duration: 1.2,
            repeat: Infinity,
            delay: i * 0.05,
            ease: 'easeInOut',
          }}
          style={{ height: '100%', originY: 1 }}
        />
      ))}
    </div>
  )
}
