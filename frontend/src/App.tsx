import { Routes, Route, Navigate } from 'react-router-dom'
import { useAuthStore } from './stores/authStore'

// Pages
import Login from './pages/auth/Login'
import Register from './pages/auth/Register'
import Dashboard from './pages/dashboard/Dashboard'
import SymptomAnalysis from './pages/analysis/SymptomAnalysis'
import ImageAnalysis from './pages/analysis/ImageAnalysis'
import LiteratureSearch from './pages/analysis/LiteratureSearch'
import RNAAnalysis from './pages/analysis/RNAAnalysis'
import KnowledgeGraph from './pages/graph/KnowledgeGraph'
import AutoMLDashboard from './pages/automl/AutoMLDashboard'

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated)

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />
  }

  return <>{children}</>
}

function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
      <Route
        path="/dashboard"
        element={
          <ProtectedRoute>
            <Dashboard />
          </ProtectedRoute>
        }
      />
      <Route
        path="/analysis/symptoms"
        element={
          <ProtectedRoute>
            <SymptomAnalysis />
          </ProtectedRoute>
        }
      />
      <Route
        path="/analysis/image"
        element={
          <ProtectedRoute>
            <ImageAnalysis />
          </ProtectedRoute>
        }
      />
      <Route
        path="/analysis/literature"
        element={
          <ProtectedRoute>
            <LiteratureSearch />
          </ProtectedRoute>
        }
      />
      <Route
        path="/analysis/rna"
        element={
          <ProtectedRoute>
            <RNAAnalysis />
          </ProtectedRoute>
        }
      />
      <Route
        path="/automl"
        element={
          <ProtectedRoute>
            <AutoMLDashboard />
          </ProtectedRoute>
        }
      />
      <Route
        path="/graph"
        element={
          <ProtectedRoute>
            <KnowledgeGraph />
          </ProtectedRoute>
        }
      />
      <Route path="/" element={<Navigate to="/dashboard" replace />} />
    </Routes>
  )
}

export default App
