import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { apiClient } from '../../services/api/client'

interface SequenceAnalysis {
  length: number
  gc_content: number
  detected_rna_type: string
  rna_type_confidence: number
  motifs_found: string[]
}

interface DiseasePrediction {
  disease: string
  disease_en: string
  icd_code: string
  probability: number
  confidence: string
  description?: string
  clinical_significance?: string
  recommendation?: string
  related_genes: string[]
}

interface RiskAssessment {
  risk_score: number
  risk_level: string
  pathogenicity: string
  pathogenicity_confidence: number
  factors: string[]
  recommendations: string[]
}

interface RNAAnalysisResponse {
  sequence_analysis: SequenceAnalysis
  disease_predictions: DiseasePrediction[]
  risk_assessment: RiskAssessment
  disclaimer: string
}

interface SampleSequence {
  id: string
  name: string
  sequence: string
  rna_type: string
  description: string
  source?: string
  expected_diseases?: string[]
}

interface SampleDataResponse {
  sequences: SampleSequence[]
  metadata: Record<string, unknown>
}

const RNA_TYPES = [
  { value: 'auto', label: '자동 감지' },
  { value: 'mRNA', label: 'mRNA (Messenger RNA)' },
  { value: 'siRNA', label: 'siRNA (Small Interfering RNA)' },
  { value: 'circRNA', label: 'circRNA (Circular RNA)' },
  { value: 'lncRNA', label: 'lncRNA (Long Non-coding RNA)' },
]

export default function RNAAnalysis() {
  const [sequence, setSequence] = useState('')
  const [rnaType, setRnaType] = useState('auto')
  const [result, setResult] = useState<RNAAnalysisResponse | null>(null)
  const [showSampleData, setShowSampleData] = useState(true)
  const [sampleFilter, setSampleFilter] = useState('')

  // Fetch sample data
  const { data: sampleData } = useQuery<SampleDataResponse>({
    queryKey: ['rna-sample-data'],
    queryFn: async () => {
      const response = await apiClient.get<SampleDataResponse>('/api/v1/rna/sample-data')
      return response.data
    },
    enabled: showSampleData,
  })

  const analysisMutation = useMutation({
    mutationFn: async (data: { sequence: string; rna_type?: string }) => {
      const response = await apiClient.analyzeRNA(data)
      return response
    },
    onSuccess: (data) => setResult(data),
  })

  const loadSampleSequence = (sample: SampleSequence) => {
    setSequence(sample.sequence)
    setRnaType(sample.rna_type || 'auto')
    setShowSampleData(false)
  }

  const handleAnalyze = () => {
    if (sequence.trim().length < 10) {
      alert('RNA 서열은 최소 10 뉴클레오타이드가 필요합니다.')
      return
    }
    analysisMutation.mutate({
      sequence: sequence.trim(),
      rna_type: rnaType === 'auto' ? undefined : rnaType,
    })
  }

  const calculateGC = (seq: string): string => {
    const normalized = seq.toUpperCase().replace(/T/g, 'U')
    const gc = (normalized.match(/[GC]/g) || []).length
    return ((gc / normalized.length) * 100).toFixed(1)
  }

  const getRiskColor = (level: string): string => {
    switch (level) {
      case 'low':
        return 'text-green-600'
      case 'moderate':
        return 'text-yellow-600'
      case 'high':
        return 'text-orange-600'
      case 'critical':
        return 'text-red-600'
      default:
        return 'text-gray-700'
    }
  }

  const getConfidenceBadge = (confidence: string): string => {
    switch (confidence) {
      case 'high':
        return 'bg-green-100 text-green-700 border-green-300'
      case 'medium':
        return 'bg-yellow-100 text-yellow-700 border-yellow-300'
      case 'low':
        return 'bg-gray-100 text-gray-700 border-gray-300'
      default:
        return 'bg-gray-100 text-gray-700 border-gray-300'
    }
  }

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="sticky top-0 z-50 metal-header px-4 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link to="/dashboard" className="text-white/70 hover:text-white transition-colors">
              ← Dashboard
            </Link>
            <h1 className="text-xl font-bold text-white">RNA 서열 분석</h1>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Section */}
          <div className="space-y-6">
            {/* Sequence Input */}
            <div className="metal-card p-6">
              <h2 className="text-lg font-semibold text-gray-800 mb-4">RNA 서열 입력</h2>

              <textarea
                value={sequence}
                onChange={(e) => setSequence(e.target.value.toUpperCase().replace(/[^AUGCT\s]/gi, ''))}
                placeholder="RNA 서열을 입력하세요 (A, U, G, C)..."
                className="w-full h-40 px-4 py-3 metal-input font-mono text-sm resize-none"
              />

              {/* Sequence Stats */}
              {sequence.length > 0 && (
                <div className="mt-3 flex gap-4 text-sm text-gray-600">
                  <span>길이: {sequence.replace(/\s/g, '').length} nt</span>
                  <span>GC 함량: {calculateGC(sequence.replace(/\s/g, ''))}%</span>
                </div>
              )}

              {/* RNA Type Selector */}
              <div className="mt-4">
                <label className="block text-sm text-gray-700 mb-2">RNA 유형</label>
                <select
                  value={rnaType}
                  onChange={(e) => setRnaType(e.target.value)}
                  className="w-full px-4 py-2 metal-select"
                >
                  {RNA_TYPES.map((type) => (
                    <option key={type.value} value={type.value}>
                      {type.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* File Upload */}
            <div className="metal-card p-6">
              <h2 className="text-lg font-semibold text-gray-800 mb-4">파일 업로드</h2>

              <div className="space-y-4">
                {/* FASTA Upload */}
                <div>
                  <label className="block text-sm text-gray-700 mb-2">FASTA 파일</label>
                  <input
                    type="file"
                    accept=".fasta,.fa,.txt"
                    onChange={(e) => {
                      const file = e.target.files?.[0]
                      if (file) {
                        const reader = new FileReader()
                        reader.onload = (event) => {
                          const text = event.target?.result as string
                          const lines = text.split('\n')
                          const seqLines = lines.filter((line) => !line.startsWith('>'))
                          setSequence(seqLines.join('').toUpperCase().replace(/[^AUGC]/gi, ''))
                        }
                        reader.readAsText(file)
                      }
                    }}
                    className="w-full text-gray-600 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-accent-cyan/20 file:text-accent-cyan hover:file:bg-accent-cyan/30 file:cursor-pointer"
                  />
                </div>

                {/* JSON Upload */}
                <div>
                  <label className="block text-sm text-gray-700 mb-2">JSON 파일</label>
                  <input
                    type="file"
                    accept=".json"
                    onChange={(e) => {
                      const file = e.target.files?.[0]
                      if (file) {
                        const reader = new FileReader()
                        reader.onload = (event) => {
                          try {
                            const json = JSON.parse(event.target?.result as string)
                            const sequences = json.sequences || []
                            if (sequences.length > 0) {
                              setSequence(sequences[0].sequence || '')
                              setRnaType(sequences[0].rna_type || 'auto')
                            }
                          } catch {
                            alert('JSON 파일 형식이 올바르지 않습니다.')
                          }
                        }
                        reader.readAsText(file)
                      }
                    }}
                    className="w-full text-gray-600 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-purple-500/20 file:text-purple-400 hover:file:bg-purple-500/30 file:cursor-pointer"
                  />
                </div>
              </div>
            </div>

            {/* Sample Data */}
            <div className="clay-card p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-800">샘플 데이터</h2>
                <button
                  onClick={() => {
                    setShowSampleData(!showSampleData)
                    if (!showSampleData) setSampleFilter('')
                  }}
                  className="text-sm text-indigo-600 hover:text-indigo-800 font-medium"
                >
                  {showSampleData ? '닫기 ▲' : '샘플 보기 ▼'}
                </button>
              </div>

              {showSampleData && sampleData && (
                <>
                  {/* RNA Type Filter */}
                  <div className="flex flex-wrap gap-2 mb-4">
                    {['전체', 'mRNA', 'siRNA', 'circRNA', 'lncRNA'].map((type) => {
                      const filterValue = type === '전체' ? '' : type
                      const isActive = sampleFilter === filterValue
                      return (
                        <button
                          key={type}
                          onClick={() => setSampleFilter(filterValue)}
                          className={`px-3 py-1 text-xs font-medium rounded-full transition-colors ${
                            isActive
                              ? 'bg-indigo-600 text-white'
                              : 'bg-indigo-100 text-indigo-700 hover:bg-indigo-200'
                          }`}
                        >
                          {type}
                        </button>
                      )
                    })}
                  </div>

                  {/* Sample List with Scrollbar */}
                  <div
                    className="space-y-2 max-h-64 overflow-y-auto pr-2"
                    style={{ scrollbarWidth: 'thin', scrollbarColor: '#a5b4fc #f3f4f6' }}
                  >
                    {sampleData.sequences
                      .filter((sample) => !sampleFilter || sample.rna_type === sampleFilter)
                      .map((sample) => (
                        <button
                          key={sample.id}
                          onClick={() => loadSampleSequence(sample)}
                          className="w-full text-left p-3 bg-white/80 hover:bg-white border border-gray-200 rounded-xl transition-all shadow-sm hover:shadow-md"
                        >
                          <div className="flex items-center justify-between">
                            <span className="font-medium text-gray-800">{sample.name}</span>
                            <span className={`px-2 py-0.5 text-xs rounded-full font-medium ${
                              sample.rna_type === 'mRNA' ? 'bg-blue-100 text-blue-700' :
                              sample.rna_type === 'siRNA' ? 'bg-green-100 text-green-700' :
                              sample.rna_type === 'circRNA' ? 'bg-purple-100 text-purple-700' :
                              'bg-orange-100 text-orange-700'
                            }`}>
                              {sample.rna_type}
                            </span>
                          </div>
                          <p className="text-xs text-gray-600 mt-1">{sample.description}</p>
                          <p className="text-xs text-gray-600 mt-1 font-mono truncate">
                            {sample.sequence.substring(0, 40)}...
                          </p>
                          {sample.expected_diseases && sample.expected_diseases.length > 0 && (
                            <div className="flex flex-wrap gap-1 mt-2">
                              {sample.expected_diseases.slice(0, 2).map((disease, idx) => (
                                <span key={idx} className="px-1.5 py-0.5 text-xs bg-red-50 text-red-600 rounded">
                                  {disease}
                                </span>
                              ))}
                            </div>
                          )}
                        </button>
                      ))}
                  </div>

                  <p className="text-xs text-gray-600 mt-3 text-center">
                    총 {sampleData.sequences.filter((s) => !sampleFilter || s.rna_type === sampleFilter).length}개 샘플 • 클릭하여 로드
                  </p>
                </>
              )}

              {!showSampleData && (
                <p className="text-sm text-gray-600">
                  샘플 RNA 서열을 로드하여 테스트할 수 있습니다. (mRNA, siRNA, circRNA, lncRNA)
                </p>
              )}
            </div>

            {/* Analyze Button */}
            <button
              onClick={handleAnalyze}
              disabled={analysisMutation.isPending || sequence.length < 10}
              className="w-full py-3 px-4 bg-accent-cyan hover:bg-accent-cyan/80 disabled:bg-accent-cyan/30 disabled:cursor-not-allowed text-black font-semibold rounded-lg transition-colors"
            >
              {analysisMutation.isPending ? '분석 중...' : 'RNA 분석'}
            </button>

            {analysisMutation.isError && (
              <div className="p-4 bg-red-100 border border-red-300 rounded-lg text-red-700">
                분석 중 오류가 발생했습니다. 다시 시도해주세요.
              </div>
            )}
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            {result ? (
              <>
                {/* Sequence Analysis */}
                <div className="metal-card p-6">
                  <h2 className="text-lg font-semibold text-gray-800 mb-4">서열 분석</h2>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <span className="text-sm text-gray-600">길이</span>
                      <p className="text-lg font-semibold text-gray-800">{result.sequence_analysis.length} nt</p>
                    </div>
                    <div>
                      <span className="text-sm text-gray-600">GC 함량</span>
                      <p className="text-lg font-semibold text-gray-800">{result.sequence_analysis.gc_content}%</p>
                    </div>
                    <div>
                      <span className="text-sm text-gray-600">RNA 유형</span>
                      <p className="text-lg font-semibold text-gray-800">{result.sequence_analysis.detected_rna_type}</p>
                    </div>
                    <div>
                      <span className="text-sm text-gray-600">유형 신뢰도</span>
                      <p className="text-lg font-semibold text-gray-800">
                        {(result.sequence_analysis.rna_type_confidence * 100).toFixed(0)}%
                      </p>
                    </div>
                  </div>

                  {result.sequence_analysis.motifs_found.length > 0 && (
                    <div className="mt-4">
                      <span className="text-sm text-gray-600">발견된 모티프</span>
                      <div className="flex flex-wrap gap-2 mt-2">
                        {result.sequence_analysis.motifs_found.map((motif, idx) => (
                          <span key={idx} className="px-2 py-1 bg-cyan-100 text-cyan-700 text-xs rounded-md">
                            {motif}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Risk Assessment */}
                <div className="metal-card p-6">
                  <h2 className="text-lg font-semibold text-gray-800 mb-4">위험도 평가</h2>

                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <span className="text-sm text-gray-600">위험 점수</span>
                      <p className={`text-3xl font-bold ${getRiskColor(result.risk_assessment.risk_level)}`}>
                        {result.risk_assessment.risk_score.toFixed(1)}
                      </p>
                    </div>
                    <div className={`px-4 py-2 rounded-lg ${getRiskColor(result.risk_assessment.risk_level)} bg-current/10`}>
                      <span className="font-semibold uppercase">{result.risk_assessment.risk_level}</span>
                    </div>
                  </div>

                  <div className="mb-4">
                    <div className="flex justify-between text-sm text-gray-600 mb-1">
                      <span>병원성</span>
                      <span>{result.risk_assessment.pathogenicity}</span>
                    </div>
                    <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${getRiskColor(result.risk_assessment.risk_level).replace('text-', 'bg-')}`}
                        style={{ width: `${result.risk_assessment.risk_score}%` }}
                      />
                    </div>
                  </div>

                  {result.risk_assessment.factors.length > 0 && (
                    <div className="mt-4">
                      <span className="text-sm font-medium text-amber-700">위험 요인</span>
                      <ul className="mt-2 space-y-1">
                        {result.risk_assessment.factors.map((factor, idx) => (
                          <li key={idx} className="text-sm text-gray-700 flex items-center gap-2">
                            <span className="w-1.5 h-1.5 bg-yellow-400 rounded-full" />
                            {factor}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {result.risk_assessment.recommendations.length > 0 && (
                    <div className="mt-4">
                      <span className="text-sm font-medium text-cyan-700">권장 사항</span>
                      <ul className="mt-2 space-y-1">
                        {result.risk_assessment.recommendations.map((rec, idx) => (
                          <li key={idx} className="text-sm text-cyan-700 flex items-center gap-2">
                            <span className="w-1.5 h-1.5 bg-accent-cyan rounded-full" />
                            {rec}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>

                {/* Disease Predictions */}
                <div className="metal-card p-6">
                  <h2 className="text-lg font-semibold text-gray-800 mb-4">질병 예측</h2>

                  <div className="space-y-4">
                    {result.disease_predictions.map((pred, idx) => (
                      <div key={idx} className="border-b border-gray-200 pb-4 last:border-0 last:pb-0">
                        <div className="flex items-start justify-between">
                          <div>
                            <p className="font-medium text-gray-800">{pred.disease}</p>
                            <p className="text-sm text-gray-600">{pred.disease_en}</p>
                            {pred.icd_code !== 'N/A' && (
                              <p className="text-xs text-gray-600 mt-1">ICD-10: {pred.icd_code}</p>
                            )}
                          </div>
                          <span className={`px-2 py-1 rounded-md text-xs font-medium border ${getConfidenceBadge(pred.confidence)}`}>
                            {pred.confidence.toUpperCase()}
                          </span>
                        </div>

                        <div className="mt-2">
                          <div className="flex justify-between text-sm text-gray-600 mb-1">
                            <span>확률</span>
                            <span>{(pred.probability * 100).toFixed(1)}%</span>
                          </div>
                          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-accent-cyan transition-all"
                              style={{ width: `${pred.probability * 100}%` }}
                            />
                          </div>
                        </div>

                        {/* Description */}
                        {pred.description && (
                          <p className="mt-3 text-sm text-gray-700 leading-relaxed">
                            {pred.description}
                          </p>
                        )}

                        {/* Clinical Significance */}
                        {pred.clinical_significance && (
                          <div className="mt-2 p-2 bg-yellow-100 border-l-2 border-yellow-500 rounded-r">
                            <p className="text-xs text-yellow-700 font-medium">임상적 중요성</p>
                            <p className="text-sm text-gray-700">{pred.clinical_significance}</p>
                          </div>
                        )}

                        {/* Recommendation */}
                        {pred.recommendation && (
                          <div className="mt-2 p-2 bg-cyan-50 border-l-2 border-cyan-500 rounded-r">
                            <p className="text-xs text-cyan-700 font-medium">권고사항</p>
                            <p className="text-sm text-gray-700">{pred.recommendation}</p>
                          </div>
                        )}

                        {/* Related Genes */}
                        {pred.related_genes.length > 0 && (
                          <div className="mt-3">
                            <p className="text-xs text-gray-600 mb-1">관련 유전자</p>
                            <div className="flex flex-wrap gap-1">
                              {pred.related_genes.map((gene, gIdx) => (
                                <span key={gIdx} className="px-2 py-0.5 bg-blue-100 text-blue-700 text-xs rounded">
                                  {gene}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Disclaimer */}
                <div className="p-4 bg-yellow-100 border border-yellow-300 rounded-lg">
                  <p className="text-sm text-yellow-800">{result.disclaimer}</p>
                </div>
              </>
            ) : (
              <div className="metal-card p-12 text-center">
                <div className="text-4xl mb-4">🧬</div>
                <h3 className="text-lg font-semibold text-gray-800 mb-2">RNA 서열을 입력하세요</h3>
                <p className="text-sm text-gray-600">
                  RNA 서열을 입력하고 분석 버튼을 클릭하면
                  <br />
                  질병 예측 및 위험도 평가 결과를 확인할 수 있습니다.
                </p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
