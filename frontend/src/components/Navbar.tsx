import { Link, NavLink } from 'react-router-dom'
import { Shield, UploadCloud, Mic, LayoutDashboard, Info } from 'lucide-react'

const links = [
  { to: '/upload',    label: 'Analyze File',  icon: UploadCloud },
  { to: '/live',      label: 'Live Detect',   icon: Mic },
  { to: '/dashboard', label: 'Dashboard',     icon: LayoutDashboard },
  { to: '/about',     label: 'About',         icon: Info },
]

export default function Navbar() {
  return (
    <nav className="sticky top-0 z-50 bg-gray-950/90 backdrop-blur border-b border-gray-800">
      <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-2.5 group">
          <span className="p-1.5 bg-brand-600 rounded-lg group-hover:bg-brand-700 transition-colors">
            <Shield size={18} className="text-white" />
          </span>
          <span className="font-bold text-white text-lg tracking-tight">
            Voice<span className="text-brand-400">Shield</span>
          </span>
        </Link>

        {/* Nav links */}
        <div className="hidden md:flex items-center gap-1">
          {links.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors
                 ${isActive
                    ? 'bg-brand-600/20 text-brand-400'
                    : 'text-gray-400 hover:text-white hover:bg-gray-800'}`
              }
            >
              <Icon size={15} />
              {label}
            </NavLink>
          ))}
        </div>

        {/* CTA */}
        <Link to="/upload" className="btn-primary text-sm hidden sm:inline-flex">
          Try Now
        </Link>
      </div>
    </nav>
  )
}
