import { useState, useEffect } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { apiClient } from '@/services/api/client'
import type {
  LiteratureSearchResponse,
  RAGQueryResponse,
  AcademicSource,
  AcademicSearchResponse,
  AcademicArticle
} from '@/types'

// Hybrid search weight constants (matching backend)
const SPARSE_WEIGHT = 0.3
const DENSE_WEIGHT = 0.7

export default function LiteratureSearch() {
  const [searchQuery, setSearchQuery] = useState('')
  const [question, setQuestion] = useState('')
  const [searchResults, setSearchResults] = useState<LiteratureSearchResponse | null>(null)
  const [ragResult, setRagResult] = useState<RAGQueryResponse | null>(null)
  const [activeTab, setActiveTab] = useState<'search' | 'qa' | 'academic'>('search')
  const [showSearchInfo, setShowSearchInfo] = useState(false)

  // Academic search state
  const [academicQuery, setAcademicQuery] = useState('')
  const [selectedSources, setSelectedSources] = useState<string[]>([])
  const [autoIngest, setAutoIngest] = useState(false)
  const [academicResults, setAcademicResults] = useState<AcademicSearchResponse | null>(null)
  const [maxResultsPerSource, setMaxResultsPerSource] = useState(10)

  // JSON viewer state
  const [selectedJsonData, setSelectedJsonData] = useState<object | null>(null)
  const [jsonCopied, setJsonCopied] = useState(false)

  // Get RAG stats
  const { data: stats } = useQuery({
    queryKey: ['rag-stats'],
    queryFn: () => apiClient.getRAGStats(),
  })

  // Get available academic sources
  const { data: sourcesData } = useQuery({
    queryKey: ['academic-sources'],
    queryFn: () => apiClient.getAcademicSources(),
  })

  // Initialize selected sources when data loads
  useEffect(() => {
    if (sourcesData?.sources && selectedSources.length === 0) {
      setSelectedSources(sourcesData.sources.map((s: AcademicSource) => s.id))
    }
  }, [sourcesData])

  // Search mutation
  const searchMutation = useMutation({
    mutationFn: (query: string) => apiClient.searchLiterature({ query, top_k: 5 }),
    onSuccess: (data) => {
      setSearchResults(data)
    },
  })

  // RAG query mutation
  const ragMutation = useMutation({
    mutationFn: (question: string) => apiClient.queryRAG({
      question,
      include_sources: true,
    }),
    onSuccess: (data) => {
      setRagResult(data)
    },
  })

  // Load sample data mutation
  const loadDataMutation = useMutation({
    mutationFn: () => apiClient.loadSampleData(),
  })

  // Academic search mutation
  const academicSearchMutation = useMutation({
    mutationFn: () => apiClient.searchAcademic({
      query: academicQuery,
      sources: selectedSources.length > 0 ? selectedSources : undefined,
      max_results_per_source: maxResultsPerSource,
      ingest: autoIngest,
    }),
    onSuccess: (data) => {
      setAcademicResults(data)
    },
  })

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (searchQuery.trim()) {
      searchMutation.mutate(searchQuery)
    }
  }

  const handleQuestion = (e: React.FormEvent) => {
    e.preventDefault()
    if (question.trim()) {
      ragMutation.mutate(question)
    }
  }

  const handleAcademicSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (academicQuery.trim() && selectedSources.length > 0) {
      academicSearchMutation.mutate()
    }
  }

  const toggleSource = (sourceId: string) => {
    setSelectedSources(prev =>
      prev.includes(sourceId)
        ? prev.filter(id => id !== sourceId)
        : [...prev, sourceId]
    )
  }

  const selectAllSources = () => {
    if (sourcesData?.sources) {
      setSelectedSources(sourcesData.sources.map((s: AcademicSource) => s.id))
    }
  }

  const deselectAllSources = () => {
    setSelectedSources([])
  }

  const getConfidenceColor = (confidence: string) => {
    switch (confidence) {
      case 'high': return 'metal-badge-green'
      case 'medium': return 'metal-badge-yellow'
      default: return 'metal-badge'
    }
  }

  // JSON viewer helper functions
  const copyJsonToClipboard = async () => {
    if (selectedJsonData) {
      await navigator.clipboard.writeText(JSON.stringify(selectedJsonData, null, 2))
      setJsonCopied(true)
      setTimeout(() => setJsonCopied(false), 2000)
    }
  }

  const selectResultForJson = (result: object) => {
    setSelectedJsonData(result)
  }

  const showAllResultsJson = () => {
    if (searchResults) {
      setSelectedJsonData(searchResults)
    } else if (academicResults) {
      setSelectedJsonData(academicResults)
    } else if (ragResult) {
      setSelectedJsonData(ragResult)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#1A1D21] to-[#15171A]">
      {/* Header */}
      <header className="metal-header">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link to="/dashboard" className="text-metal-text-muted hover:text-metal-text-light transition-colors">
              ← 대시보드
            </Link>
            <h1 className="text-xl font-bold text-metal-text-light">의학 문헌 검색</h1>
          </div>
          {stats && (
            <div className="text-sm text-metal-text-muted flex items-center gap-3">
              <span>문헌 수: {stats.document_count}개</span>
              {stats.search_method && (
                <span className="px-2 py-1 rounded text-xs" style={{ background: 'linear-gradient(180deg, #9B59B6 0%, #8E44AD 100%)', color: '#FFF' }}>
                  {stats.search_method}
                </span>
              )}
            </div>
          )}
        </div>
      </header>

      <main className="max-w-full mx-auto px-4 py-8">
        <div className="flex gap-6">
          {/* Left Panel - Search Content */}
          <div className="flex-1 min-w-0">
            {/* Tabs */}
            <div className="flex gap-4 mb-6">
              <button
                onClick={() => setActiveTab('search')}
                className={`px-4 py-2 rounded-metal font-medium transition-all ${
                  activeTab === 'search'
                    ? 'metal-btn'
                    : 'metal-btn-secondary'
                }`}
              >
                문헌 검색
              </button>
              <button
                onClick={() => setActiveTab('qa')}
                className={`px-4 py-2 rounded-metal font-medium transition-all ${
                  activeTab === 'qa'
                    ? 'metal-btn'
                    : 'metal-btn-secondary'
                }`}
              >
                질문 답변 (RAG)
              </button>
              <button
                onClick={() => setActiveTab('academic')}
                className={`px-4 py-2 rounded-metal font-medium transition-all ${
                  activeTab === 'academic'
                    ? 'metal-badge-green'
                    : 'metal-btn-secondary'
                }`}
              >
                학술 검색 (외부 API)
              </button>
            </div>

            {/* Load Sample Data */}
            {stats?.document_count === 0 && (
              <div className="p-4 rounded-metal mb-6"
                style={{
                  background: 'linear-gradient(180deg, #2A3A4A 0%, #223344 100%)',
                  borderTop: '1px solid rgba(79, 195, 247, 0.2)',
                  borderLeft: '3px solid #4FC3F7'
                }}>
                <p className="text-accent-cyan mb-2">문헌 데이터베이스가 비어있습니다.</p>
                <button
                  onClick={() => loadDataMutation.mutate()}
                  disabled={loadDataMutation.isPending}
                  className="px-4 py-2 metal-btn disabled:opacity-50"
                >
                  {loadDataMutation.isPending ? '로딩 중...' : '샘플 데이터 로드'}
                </button>
              </div>
            )}

            {/* Search Tab */}
            {activeTab === 'search' && (
              <div className="space-y-6 max-w-4xl">
                {/* Hybrid Search Info Panel */}
                <div className="rounded-metal p-4"
                  style={{
                    background: 'linear-gradient(180deg, #2A2F37 0%, #252930 100%)',
                    border: '1px solid rgba(155, 89, 182, 0.3)'
                  }}>
                  <button
                    onClick={() => setShowSearchInfo(!showSearchInfo)}
                    className="w-full flex items-center justify-between text-left"
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-lg">🔍</span>
                      <span className="font-medium" style={{ color: '#B39DDB' }}>Qdrant 하이브리드 검색 시스템</span>
                      <span className="text-xs px-2 py-0.5 rounded" style={{ background: 'rgba(155, 89, 182, 0.3)', color: '#B39DDB' }}>
                        Sparse {(SPARSE_WEIGHT * 100).toFixed(0)}% + Dense {(DENSE_WEIGHT * 100).toFixed(0)}%
                      </span>
                    </div>
                    <span style={{ color: '#B39DDB' }}>{showSearchInfo ? '▲' : '▼'}</span>
                  </button>

                  {showSearchInfo && (
                    <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
                      <div className="rounded-metal p-3"
                        style={{
                          background: 'linear-gradient(180deg, #3A2F2A 0%, #2E2520 100%)',
                          border: '1px solid rgba(230, 126, 34, 0.3)'
                        }}>
                        <h4 className="font-medium text-orange-400 mb-2">🔤 Sparse (BM25) - {(SPARSE_WEIGHT * 100).toFixed(0)}%</h4>
                        <p className="text-metal-text-muted text-xs">
                          키워드 기반 검색. 정확한 용어 매칭에 강점.
                          한국어 포함 다국어 지원. TF-IDF 기반 가중치.
                        </p>
                      </div>
                      <div className="rounded-metal p-3"
                        style={{
                          background: 'linear-gradient(180deg, #2A3A4A 0%, #223344 100%)',
                          border: '1px solid rgba(79, 195, 247, 0.3)'
                        }}>
                        <h4 className="font-medium text-accent-cyan mb-2">📊 Dense (BioBERT) - {(DENSE_WEIGHT * 100).toFixed(0)}%</h4>
                        <p className="text-metal-text-muted text-xs">
                          의미 기반 검색. 문맥과 개념 이해에 강점.
                          유사 개념이나 관련 문헌 검색에 효과적.
                        </p>
                      </div>
                    </div>
                  )}
                </div>

                <form onSubmit={handleSearch} className="metal-card p-6">
                  <h2 className="text-lg font-semibold text-metal-text-light mb-4">의학 문헌 검색</h2>
                  <div className="flex gap-4">
                    <input
                      type="text"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      placeholder="검색어를 입력하세요 (예: 폐렴 치료 항생제)"
                      className="flex-1 px-4 py-2 metal-input"
                    />
                    <button
                      type="submit"
                      disabled={searchMutation.isPending || !searchQuery.trim()}
                      className="px-6 py-2 metal-btn disabled:opacity-50"
                    >
                      {searchMutation.isPending ? '검색 중...' : '검색'}
                    </button>
                  </div>
                </form>

                {/* Search Results */}
                {searchResults && (
                  <div className="metal-card p-6">
                    <h2 className="text-lg font-semibold text-metal-text-light mb-4">
                      검색 결과 ({searchResults.total_found}건)
                    </h2>

                    {searchResults.results.length === 0 ? (
                      <p className="text-metal-text-muted">검색 결과가 없습니다.</p>
                    ) : (
                      <div className="space-y-4">
                        {searchResults.results.map((result, index) => (
                          <div
                            key={index}
                            className="rounded-metal p-4 cursor-pointer transition-all hover:shadow-metal"
                            style={{
                              background: 'linear-gradient(180deg, #2A2F37 0%, #252930 100%)',
                              borderTop: '1px solid rgba(255,255,255,0.08)',
                              borderBottom: '1px solid rgba(0,0,0,0.3)'
                            }}
                            onClick={() => selectResultForJson(result)}
                          >
                            <div className="flex justify-between items-start mb-2">
                              <h3 className="font-medium text-accent-cyan">
                                {result.metadata.title || '제목 없음'}
                              </h3>
                              <span className="text-sm font-semibold whitespace-nowrap ml-2" style={{ color: '#B39DDB' }}>
                                종합: {(result.score * 100).toFixed(1)}%
                              </span>
                            </div>

                            {/* Hybrid Score Breakdown */}
                            {result.hybrid_score && (
                              <div className="mb-3 p-3 rounded-metal" style={{ background: 'rgba(0,0,0,0.2)' }}>
                                <div className="flex gap-4 text-xs">
                                  {/* Sparse Score */}
                                  <div className="flex-1">
                                    <div className="flex justify-between mb-1">
                                      <span className="text-orange-400 font-medium">Sparse (BM25)</span>
                                      <span className="text-orange-400">{(result.hybrid_score.sparse_score * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="h-2 rounded-full overflow-hidden" style={{ background: 'rgba(0,0,0,0.3)' }}>
                                      <div
                                        className="h-full rounded-full transition-all"
                                        style={{ width: `${result.hybrid_score.sparse_score * 100}%`, background: 'linear-gradient(180deg, #E67E22 0%, #D35400 100%)' }}
                                      />
                                    </div>
                                  </div>
                                  {/* Dense Score */}
                                  <div className="flex-1">
                                    <div className="flex justify-between mb-1">
                                      <span className="text-accent-cyan font-medium">Dense (BioBERT)</span>
                                      <span className="text-accent-cyan">{(result.hybrid_score.dense_score * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="h-2 rounded-full overflow-hidden" style={{ background: 'rgba(0,0,0,0.3)' }}>
                                      <div
                                        className="h-full metal-progress-bar rounded-full transition-all"
                                        style={{ width: `${result.hybrid_score.dense_score * 100}%` }}
                                      />
                                    </div>
                                  </div>
                                </div>
                                <div className="mt-2 text-xs text-metal-text-muted text-center">
                                  가중치: Sparse {(SPARSE_WEIGHT * 100).toFixed(0)}% + Dense {(DENSE_WEIGHT * 100).toFixed(0)}%
                                </div>
                              </div>
                            )}

                            <div className="text-sm text-metal-text-mid mb-3 leading-relaxed whitespace-pre-wrap p-3 rounded-metal" style={{ background: 'rgba(0,0,0,0.2)' }}>
                              {result.content}
                            </div>
                            <div className="text-xs text-metal-text-muted border-t border-white/10 pt-2">
                              {result.metadata.authors && <span className="font-medium">{result.metadata.authors}</span>}
                              {result.metadata.year && <span> • {result.metadata.year}</span>}
                              {result.metadata.journal && <span> • {result.metadata.journal}</span>}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* QA Tab */}
            {activeTab === 'qa' && (
              <div className="space-y-6">
                <form onSubmit={handleQuestion} className="metal-card p-6 max-w-4xl">
                  <h2 className="text-lg font-semibold text-metal-text-light mb-4">의학 질문하기</h2>
                  <p className="text-sm text-metal-text-muted mb-4">
                    의학 문헌 + 지식 그래프를 기반으로 질문에 답변합니다.
                  </p>
                  <textarea
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="질문을 입력하세요 (예: 폐렴의 일반적인 치료법은 무엇인가요?)"
                    className="w-full px-4 py-2 metal-input h-24 resize-none"
                  />
                  <button
                    type="submit"
                    disabled={ragMutation.isPending || !question.trim()}
                    className="mt-4 px-6 py-2 metal-btn disabled:opacity-50"
                  >
                    {ragMutation.isPending ? '답변 생성 중...' : '질문하기'}
                  </button>
                </form>

                {/* RAG Result */}
                {ragResult && (
                  <div className="max-w-4xl">
                    <div className="space-y-4">
                      {/* Answer */}
                      <div className="metal-card p-6">
                        <div className="flex justify-between items-center mb-4">
                          <h2 className="text-lg font-semibold text-metal-text-light">답변</h2>
                          <span className={`text-xs px-2 py-1 rounded-metal-sm ${getConfidenceColor(ragResult.confidence)}`}>
                            신뢰도: {ragResult.confidence === 'high' ? '높음' : ragResult.confidence === 'medium' ? '중간' : '낮음'}
                          </span>
                        </div>
                        <div className="prose prose-sm max-w-none">
                          <pre className="whitespace-pre-wrap font-sans text-metal-text-mid p-4 rounded-metal text-sm max-h-96 overflow-y-auto" style={{ background: 'rgba(0,0,0,0.2)' }}>
                            {ragResult.answer}
                          </pre>
                        </div>
                      </div>

                      {/* Sources */}
                      {ragResult.sources.length > 0 && (
                        <div className="metal-card p-6">
                          <div className="flex items-center justify-between mb-4">
                            <h2 className="text-lg font-semibold text-metal-text-light">참고 문헌</h2>
                            <span className="text-xs px-2 py-1 rounded" style={{ background: 'rgba(155, 89, 182, 0.3)', color: '#B39DDB' }}>
                              Hybrid Search (3:7)
                            </span>
                          </div>
                          <div className="space-y-3 max-h-64 overflow-y-auto">
                            {ragResult.sources.map((source, index) => (
                              <div
                                key={index}
                                className="rounded-metal p-3 cursor-pointer transition-all hover:shadow-metal"
                                style={{
                                  background: 'linear-gradient(180deg, #2A2F37 0%, #252930 100%)',
                                  borderTop: '1px solid rgba(255,255,255,0.05)'
                                }}
                                onClick={() => selectResultForJson(source)}
                              >
                                <div className="flex justify-between items-start mb-2">
                                  <div>
                                    <p className="font-medium text-sm text-metal-text-light">[{index + 1}] {source.title}</p>
                                    <p className="text-xs text-metal-text-muted">
                                      {source.authors && <span>{source.authors}</span>}
                                      {source.year && <span> ({source.year})</span>}
                                    </p>
                                  </div>
                                  <span className="text-sm font-semibold" style={{ color: '#B39DDB' }}>
                                    {(source.relevance * 100).toFixed(1)}%
                                  </span>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Disclaimer */}
                    <div className="p-4 rounded-metal mt-4"
                      style={{
                        background: 'linear-gradient(180deg, #3D3A28 0%, #2E2B1F 100%)',
                        borderLeft: '3px solid #B7950B',
                        borderTop: '1px solid rgba(255,255,255,0.08)'
                      }}>
                      <p className="text-sm" style={{ color: '#D4AC0D' }}>{ragResult.disclaimer}</p>
                    </div>
                  </div>
                )}

                {!ragResult && !ragMutation.isPending && (
                  <div className="metal-card p-8 text-center text-metal-text-muted max-w-4xl">
                    의학 관련 질문을 입력하세요
                  </div>
                )}
              </div>
            )}

            {/* Academic Search Tab */}
            {activeTab === 'academic' && (
              <div className="space-y-6">
                {/* Info Panel */}
                <div className="rounded-metal p-4"
                  style={{
                    background: 'linear-gradient(180deg, #2A3D2A 0%, #1F2E1F 100%)',
                    border: '1px solid rgba(38, 194, 129, 0.3)'
                  }}>
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-lg">📚</span>
                    <span className="font-medium text-accent-green">다중 소스 학술 검색</span>
                  </div>
                  <p className="text-sm text-metal-text-mid">
                    PubMed, Semantic Scholar, CrossRef, OpenAlex, KoreaMed 등 여러 학술 데이터베이스에서 동시에 검색합니다.
                    검색된 논문은 VectorDB에 자동 저장하여 RAG 검색에 활용할 수 있습니다.
                  </p>
                </div>

                {/* Search Form */}
                <form onSubmit={handleAcademicSearch} className="metal-card p-6">
                  <h2 className="text-lg font-semibold text-metal-text-light mb-4">학술 논문 검색</h2>

                  {/* Query Input */}
                  <div className="mb-4">
                    <label className="block text-sm font-medium text-metal-text-mid mb-2">검색어</label>
                    <input
                      type="text"
                      value={academicQuery}
                      onChange={(e) => setAcademicQuery(e.target.value)}
                      placeholder="검색어를 입력하세요 (예: COVID-19 treatment, 폐렴 치료)"
                      className="w-full px-4 py-2 metal-input"
                    />
                  </div>

                  {/* Source Selection */}
                  <div className="mb-4">
                    <div className="flex items-center justify-between mb-2">
                      <label className="block text-sm font-medium text-metal-text-mid">검색 소스</label>
                      <div className="flex gap-2">
                        <button
                          type="button"
                          onClick={selectAllSources}
                          className="text-xs text-accent-cyan hover:text-accent-cyan-light transition-colors"
                        >
                          전체 선택
                        </button>
                        <span className="text-metal-text-muted">|</span>
                        <button
                          type="button"
                          onClick={deselectAllSources}
                          className="text-xs text-metal-text-muted hover:text-metal-text-light transition-colors"
                        >
                          전체 해제
                        </button>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
                      {sourcesData?.sources?.map((source: AcademicSource) => (
                        <label
                          key={source.id}
                          className={`flex items-center gap-2 p-3 rounded-metal cursor-pointer transition-all ${
                            selectedSources.includes(source.id)
                              ? ''
                              : ''
                          }`}
                          style={{
                            background: selectedSources.includes(source.id)
                              ? 'linear-gradient(180deg, #2A3D2A 0%, #1F2E1F 100%)'
                              : 'linear-gradient(180deg, #2A2F37 0%, #252930 100%)',
                            border: selectedSources.includes(source.id)
                              ? '1px solid rgba(38, 194, 129, 0.3)'
                              : '1px solid rgba(255,255,255,0.05)'
                          }}
                        >
                          <input
                            type="checkbox"
                            checked={selectedSources.includes(source.id)}
                            onChange={() => toggleSource(source.id)}
                            className="rounded text-accent-green"
                          />
                          <div>
                            <p className="text-sm font-medium text-metal-text-light">{source.name}</p>
                            <p className="text-xs text-metal-text-muted">{source.description}</p>
                          </div>
                        </label>
                      ))}
                    </div>
                    {selectedSources.length === 0 && (
                      <p className="text-sm text-red-400 mt-2">최소 1개 이상의 소스를 선택하세요.</p>
                    )}
                  </div>

                  {/* Options */}
                  <div className="flex flex-wrap items-center gap-6 mb-4">
                    {/* Max Results */}
                    <div className="flex items-center gap-2">
                      <label className="text-sm text-metal-text-muted">소스당 최대 결과:</label>
                      <select
                        value={maxResultsPerSource}
                        onChange={(e) => setMaxResultsPerSource(Number(e.target.value))}
                        className="px-3 py-1 metal-select text-sm"
                      >
                        <option value={5}>5개</option>
                        <option value={10}>10개</option>
                        <option value={20}>20개</option>
                        <option value={50}>50개</option>
                      </select>
                    </div>

                    {/* Auto Ingest Toggle */}
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={autoIngest}
                        onChange={(e) => setAutoIngest(e.target.checked)}
                        className="rounded text-accent-green"
                      />
                      <span className="text-sm text-metal-text-muted">VectorDB에 자동 저장</span>
                      <span className="text-xs px-2 py-0.5 metal-badge-green rounded">RAG 활용</span>
                    </label>
                  </div>

                  {/* Submit Button */}
                  <button
                    type="submit"
                    disabled={academicSearchMutation.isPending || !academicQuery.trim() || selectedSources.length === 0}
                    className="px-6 py-2 metal-badge-green rounded-metal font-medium disabled:opacity-50 transition-all"
                  >
                    {academicSearchMutation.isPending ? (
                      <span className="flex items-center gap-2">
                        <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                        </svg>
                        검색 중...
                      </span>
                    ) : (
                      `${selectedSources.length}개 소스에서 검색`
                    )}
                  </button>
                </form>

                {/* Results */}
                {academicResults && (
                  <div className="metal-card p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h2 className="text-lg font-semibold text-metal-text-light">
                        검색 결과 ({academicResults.total_found}건)
                      </h2>
                      <div className="flex items-center gap-3 text-sm">
                        <span className="text-metal-text-muted">
                          검색 소스: {academicResults.sources_searched.join(', ')}
                        </span>
                        {academicResults.ingested > 0 && (
                          <span className="px-2 py-1 metal-badge-green rounded">
                            {academicResults.ingested}건 VectorDB 저장됨
                          </span>
                        )}
                      </div>
                    </div>

                    {academicResults.total_found === 0 ? (
                      <p className="text-metal-text-muted">검색 결과가 없습니다.</p>
                    ) : (
                      <div className="space-y-6">
                        {Object.entries(academicResults.results).map(([source, articles]) => (
                          articles.length > 0 && (
                            <div key={source} className="rounded-metal overflow-hidden" style={{ border: '1px solid rgba(255,255,255,0.05)' }}>
                              <div className="px-4 py-2 flex items-center justify-between"
                                style={{ background: 'linear-gradient(180deg, #2C3036 0%, #23272B 100%)' }}>
                                <h3 className="font-medium text-metal-text-light">{source}</h3>
                                <span className="text-sm text-metal-text-muted">{articles.length}건</span>
                              </div>
                              <div className="divide-y divide-white/5">
                                {(articles as AcademicArticle[]).map((article, index) => (
                                  <div
                                    key={`${source}-${index}`}
                                    className="p-4 cursor-pointer transition-all hover:bg-white/5"
                                    onClick={() => selectResultForJson(article)}
                                  >
                                    <h4 className="font-medium text-accent-cyan mb-2">{article.title}</h4>
                                    <p className="text-sm text-metal-text-mid mb-2 line-clamp-3">{article.abstract}</p>
                                    <div className="flex flex-wrap items-center gap-3 text-xs text-metal-text-muted">
                                      <span>{article.authors}</span>
                                      {article.year && <span>• {article.year}</span>}
                                      {article.journal && <span>• {article.journal}</span>}
                                      {article.pmid && (
                                        <a
                                          href={`https://pubmed.ncbi.nlm.nih.gov/${article.pmid}`}
                                          target="_blank"
                                          rel="noopener noreferrer"
                                          className="text-accent-cyan hover:text-accent-cyan-light"
                                        >
                                          PMID: {article.pmid}
                                        </a>
                                      )}
                                      {article.doi && (
                                        <a
                                          href={`https://doi.org/${article.doi}`}
                                          target="_blank"
                                          rel="noopener noreferrer"
                                          className="text-accent-cyan hover:text-accent-cyan-light"
                                        >
                                          DOI
                                        </a>
                                      )}
                                    </div>
                                    {article.keywords.length > 0 && (
                                      <div className="mt-2 flex flex-wrap gap-1">
                                        {article.keywords.slice(0, 5).map((kw, i) => (
                                          <span key={i} className="px-2 py-0.5 rounded text-xs"
                                            style={{ background: 'rgba(255,255,255,0.05)', color: '#A8B0BA' }}>
                                            {kw}
                                          </span>
                                        ))}
                                      </div>
                                    )}
                                  </div>
                                ))}
                              </div>
                            </div>
                          )
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {!academicResults && !academicSearchMutation.isPending && (
                  <div className="metal-card p-8 text-center text-metal-text-muted">
                    검색할 키워드를 입력하고 검색 소스를 선택하세요
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Right Panel - JSON Metadata Viewer */}
          <div className="w-80 flex-shrink-0">
            <div className="metal-card sticky top-4">
              <div className="flex items-center justify-between p-4 border-b border-white/5">
                <h3 className="font-semibold text-metal-text-light text-sm">JSON 메타데이터</h3>
                <div className="flex gap-1">
                  <button
                    onClick={showAllResultsJson}
                    disabled={!searchResults && !academicResults && !ragResult}
                    className="px-2 py-1 text-xs metal-btn disabled:opacity-50"
                  >
                    전체
                  </button>
                  <button
                    onClick={copyJsonToClipboard}
                    disabled={!selectedJsonData}
                    className="px-2 py-1 text-xs metal-btn-secondary disabled:opacity-50"
                  >
                    {jsonCopied ? '복사됨!' : '복사'}
                  </button>
                  <button
                    onClick={() => setSelectedJsonData(null)}
                    disabled={!selectedJsonData}
                    className="px-2 py-1 text-xs metal-btn-danger disabled:opacity-50"
                  >
                    지우기
                  </button>
                </div>
              </div>
              <div className="p-4">
                {selectedJsonData ? (
                  <pre className="text-xs p-4 rounded-metal overflow-auto max-h-[calc(100vh-250px)] font-mono metal-json-viewer">
                    {JSON.stringify(selectedJsonData, null, 2)}
                  </pre>
                ) : (
                  <div className="text-center text-metal-text-muted py-12">
                    <div className="text-4xl mb-2 opacity-30">{ }</div>
                    <p className="text-sm">검색 결과를 클릭하면</p>
                    <p className="text-sm">JSON 형식으로 볼 수 있습니다</p>
                  </div>
                )}
              </div>
              {selectedJsonData && (
                <div className="px-4 pb-4 text-xs text-metal-text-muted border-t border-white/5 pt-3">
                  <div className="flex justify-between">
                    <span>데이터 크기:</span>
                    <span>{JSON.stringify(selectedJsonData).length.toLocaleString()} bytes</span>
                  </div>
                  <div className="flex justify-between mt-1">
                    <span>키 개수:</span>
                    <span>{Object.keys(selectedJsonData).length}개</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
