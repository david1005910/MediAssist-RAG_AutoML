import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { useAuthStore } from '@/stores/authStore'
import { apiClient } from '@/services/api/client'

export default function Dashboard() {
  const user = useAuthStore((state) => state.user)
  const logout = useAuthStore((state) => state.logout)

  // Check API health
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: () => apiClient.healthCheck(),
    retry: false,
  })

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="metal-header px-4 py-4">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-bold text-white">MediAssist AI</h1>
            {health && (
              <span className="flex items-center gap-2 text-xs metal-badge metal-badge-green">
                <span className="w-2 h-2 rounded-full metal-status-online"></span>
                API 연결됨
              </span>
            )}
          </div>
          <div className="flex items-center gap-4">
            <span className="text-sm text-white/70">{user?.name || '사용자'}</span>
            <button
              onClick={logout}
              className="text-sm px-3 py-1.5 metal-btn"
            >
              로그아웃
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        <h2 className="text-2xl font-bold mb-6 text-gray-800">진단 보조 도구</h2>

        {/* Feature Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {/* Symptom Analysis */}
          <Link
            to="/analysis/symptoms"
            className="block p-6 metal-card hover:shadow-metal-lg transition-all group"
            style={{ borderLeft: '3px solid #4FC3F7' }}
          >
            <div className="text-3xl mb-3 opacity-80 group-hover:opacity-100 transition-opacity">🩺</div>
            <h3 className="text-lg font-semibold" style={{ color: '#1F2937' }}>증상 분석</h3>
            <p className="mt-2 text-sm text-gray-600">
              환자 증상을 입력하여 AI 기반 진단 보조 분석을 수행합니다.
            </p>
            <div className="mt-4 text-xs text-cyan-600">
              BioBERT + XGBoost 앙상블
            </div>
          </Link>

          {/* Image Analysis */}
          <Link
            to="/analysis/image"
            className="block p-6 metal-card hover:shadow-metal-lg transition-all group"
            style={{ borderLeft: '3px solid #26C281' }}
          >
            <div className="text-3xl mb-3 opacity-80 group-hover:opacity-100 transition-opacity">🔬</div>
            <h3 className="text-lg font-semibold" style={{ color: '#1F2937' }}>이미지 분석</h3>
            <p className="mt-2 text-sm text-gray-600">
              흉부 X-ray를 업로드하여 AI 분석 및 Grad-CAM 시각화를 수행합니다.
            </p>
            <div className="mt-4 text-xs text-green-600">
              DenseNet121 + Grad-CAM
            </div>
          </Link>

          {/* Literature Search */}
          <Link
            to="/analysis/literature"
            className="block p-6 metal-card hover:shadow-metal-lg transition-all group"
            style={{ borderLeft: '3px solid #9B59B6' }}
          >
            <div className="text-3xl mb-3 opacity-80 group-hover:opacity-100 transition-opacity">📚</div>
            <h3 className="text-lg font-semibold" style={{ color: '#1F2937' }}>문헌 검색</h3>
            <p className="mt-2 text-sm text-gray-600">
              의학 문헌을 검색하고 RAG 기반 질의응답을 수행합니다.
            </p>
            <div className="mt-4 text-xs text-purple-600">
              ChromaDB + Cross-encoder
            </div>
          </Link>

          {/* Knowledge Graph */}
          <Link
            to="/graph"
            className="block p-6 metal-card hover:shadow-metal-lg transition-all group"
            style={{ borderLeft: '3px solid #E67E22' }}
          >
            <div className="text-3xl mb-3 opacity-80 group-hover:opacity-100 transition-opacity">🕸️</div>
            <h3 className="text-lg font-semibold" style={{ color: '#1F2937' }}>지식 그래프</h3>
            <p className="mt-2 text-sm text-gray-600">
              의료 지식 그래프를 탐색하고 질환-증상-치료 관계를 시각화합니다.
            </p>
            <div className="mt-4 text-xs text-orange-600">
              Neo4j Graph Database
            </div>
          </Link>

          {/* RNA Analysis */}
          <Link
            to="/analysis/rna"
            className="block p-6 metal-card hover:shadow-metal-lg transition-all group"
            style={{ borderLeft: '3px solid #00BCD4' }}
          >
            <div className="text-3xl mb-3 opacity-80 group-hover:opacity-100 transition-opacity">🧬</div>
            <h3 className="text-lg font-semibold" style={{ color: '#1F2937' }}>RNA 분석</h3>
            <p className="mt-2 text-sm text-gray-600">
              RNA 서열을 분석하여 질병 예측 및 위험도 평가를 수행합니다.
            </p>
            <div className="mt-4 text-xs text-cyan-600">
              BERT + Hybrid N-gram
            </div>
          </Link>

          {/* AutoML */}
          <Link
            to="/automl"
            className="block p-6 metal-card hover:shadow-metal-lg transition-all group"
            style={{ borderLeft: '3px solid #FF5722' }}
          >
            <div className="text-3xl mb-3 opacity-80 group-hover:opacity-100 transition-opacity">🤖</div>
            <h3 className="text-lg font-semibold" style={{ color: '#1F2937' }}>AutoML</h3>
            <p className="mt-2 text-sm text-gray-600">
              자동 하이퍼파라미터 최적화 및 모델 선택을 수행합니다.
            </p>
            <div className="mt-4 text-xs text-orange-600">
              Optuna + Ensemble
            </div>
          </Link>
        </div>

        {/* System Status */}
        <div className="metal-card p-6 mb-8">
          <h3 className="text-lg font-semibold mb-4" style={{ color: '#1F2937' }}>시스템 상태</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 rounded-metal-sm"
              style={{ background: 'linear-gradient(180deg, #252930 0%, #1F2328 100%)' }}>
              <div className="text-2xl font-bold text-accent-cyan">3</div>
              <div className="text-sm text-gray-600">분석 모듈</div>
            </div>
            <div className="text-center p-4 rounded-metal-sm"
              style={{ background: 'linear-gradient(180deg, #252930 0%, #1F2328 100%)' }}>
              <div className="text-2xl font-bold text-accent-green">7</div>
              <div className="text-sm text-gray-600">검출 가능 질환</div>
            </div>
            <div className="text-center p-4 rounded-metal-sm"
              style={{ background: 'linear-gradient(180deg, #252930 0%, #1F2328 100%)' }}>
              <div className="text-2xl font-bold" style={{ color: '#B39DDB' }}>8</div>
              <div className="text-sm text-gray-600">문헌 DB</div>
            </div>
            <div className="text-center p-4 rounded-metal-sm"
              style={{ background: 'linear-gradient(180deg, #252930 0%, #1F2328 100%)' }}>
              <div className={`text-2xl font-bold ${health ? 'text-accent-green' : 'text-red-400'}`}>
                {health ? 'ON' : 'OFF'}
              </div>
              <div className="text-sm text-gray-600">API 상태</div>
            </div>
          </div>
        </div>

        {/* Disclaimer */}
        <div className="p-4 rounded-metal"
          style={{
            background: 'linear-gradient(180deg, #3D3A28 0%, #2E2B1F 100%)',
            borderLeft: '3px solid #B7950B',
            borderTop: '1px solid rgba(255,255,255,0.08)'
          }}>
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5" style={{ color: '#D4AC0D' }} viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium" style={{ color: '#F4D03F' }}>의료 면책 조항</h3>
              <p className="mt-1 text-sm" style={{ color: '#D4AC0D' }}>
                이 시스템의 모든 분석 결과는 참고용이며, 최종 진단 및 치료 결정은 반드시 자격을 갖춘 의료 전문가가 수행해야 합니다.
                AI 분석 결과를 유일한 진단 근거로 사용하지 마십시오.
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
