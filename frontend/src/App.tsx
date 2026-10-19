import { Routes, Route } from 'react-router-dom'
import Navbar from '@/components/Navbar'
import Footer from '@/components/Footer'
import HomePage from '@/pages/HomePage'
import UploadPage from '@/pages/UploadPage'
import LivePage from '@/pages/LivePage'
import DashboardPage from '@/pages/DashboardPage'
import AboutPage from '@/pages/AboutPage'

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1">
        <Routes>
          <Route path="/"          element={<HomePage />} />
          <Route path="/upload"    element={<UploadPage />} />
          <Route path="/live"      element={<LivePage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/about"     element={<AboutPage />} />
        </Routes>
      </main>
      <Footer />
    </div>
  )
}
