import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
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

  const analysisMutation = useMutation({
    mutationFn: async (data: { sequence: string; rna_type?: string }) => {
      const response = await apiClient.analyzeRNA(data)
      return response
    },
    onSuccess: (data) => setResult(data),
  })

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
        return 'text-green-400'
      case 'moderate':
        return 'text-yellow-400'
      case 'high':
        return 'text-orange-400'
      case 'critical':
        return 'text-red-400'
      default:
        return 'text-metal-text-mid'
    }
  }

  const getConfidenceBadge = (confidence: string): string => {
    switch (confidence) {
      case 'high':
        return 'bg-green-500/20 text-green-400 border-green-500/30'
      case 'medium':
        return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
      case 'low':
        return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
      default:
        return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#1A1D21] to-[#15171A]">
      {/* Header */}
      <header className="sticky top-0 z-50 backdrop-blur-xl bg-[#1A1D21]/80 border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link to="/dashboard" className="text-metal-text-muted hover:text-metal-text-light transition-colors">
              ← Dashboard
            </Link>
            <h1 className="text-xl font-bold text-metal-text-light">RNA 서열 분석</h1>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Section */}
          <div className="space-y-6">
            {/* Sequence Input */}
            <div className="bg-gradient-to-b from-[#2A2F37] to-[#252930] rounded-xl border border-white/10 p-6">
              <h2 className="text-lg font-semibold text-metal-text-light mb-4">RNA 서열 입력</h2>

              <textarea
                value={sequence}
                onChange={(e) => setSequence(e.target.value.toUpperCase().replace(/[^AUGCT\s]/gi, ''))}
                placeholder="RNA 서열을 입력하세요 (A, U, G, C)..."
                className="w-full h-40 px-4 py-3 bg-[#1A1D21] border border-white/10 rounded-lg text-metal-text-light font-mono text-sm placeholder-metal-text-muted focus:outline-none focus:ring-2 focus:ring-accent-cyan/50 resize-none"
              />

              {/* Sequence Stats */}
              {sequence.length > 0 && (
                <div className="mt-3 flex gap-4 text-sm text-metal-text-muted">
                  <span>길이: {sequence.replace(/\s/g, '').length} nt</span>
                  <span>GC 함량: {calculateGC(sequence.replace(/\s/g, ''))}%</span>
                </div>
              )}

              {/* RNA Type Selector */}
              <div className="mt-4">
                <label className="block text-sm text-metal-text-mid mb-2">RNA 유형</label>
                <select
                  value={rnaType}
                  onChange={(e) => setRnaType(e.target.value)}
                  className="w-full px-4 py-2 bg-[#1A1D21] border border-white/10 rounded-lg text-metal-text-light focus:outline-none focus:ring-2 focus:ring-accent-cyan/50"
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
            <div className="bg-gradient-to-b from-[#2A2F37] to-[#252930] rounded-xl border border-white/10 p-6">
              <h2 className="text-lg font-semibold text-metal-text-light mb-4">FASTA 파일 업로드</h2>
              <input
                type="file"
                accept=".fasta,.fa,.txt"
                onChange={(e) => {
                  const file = e.target.files?.[0]
                  if (file) {
                    const reader = new FileReader()
                    reader.onload = (event) => {
                      const text = event.target?.result as string
                      // Parse FASTA - skip header lines starting with >
                      const lines = text.split('\n')
                      const seqLines = lines.filter((line) => !line.startsWith('>'))
                      setSequence(seqLines.join('').toUpperCase().replace(/[^AUGC]/gi, ''))
                    }
                    reader.readAsText(file)
                  }
                }}
                className="w-full text-metal-text-muted file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-accent-cyan/20 file:text-accent-cyan hover:file:bg-accent-cyan/30 file:cursor-pointer"
              />
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
              <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400">
                분석 중 오류가 발생했습니다. 다시 시도해주세요.
              </div>
            )}
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            {result ? (
              <>
                {/* Sequence Analysis */}
                <div className="bg-gradient-to-b from-[#2A2F37] to-[#252930] rounded-xl border border-white/10 p-6">
                  <h2 className="text-lg font-semibold text-metal-text-light mb-4">서열 분석</h2>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <span className="text-sm text-metal-text-muted">길이</span>
                      <p className="text-lg font-semibold text-metal-text-light">{result.sequence_analysis.length} nt</p>
                    </div>
                    <div>
                      <span className="text-sm text-metal-text-muted">GC 함량</span>
                      <p className="text-lg font-semibold text-metal-text-light">{result.sequence_analysis.gc_content}%</p>
                    </div>
                    <div>
                      <span className="text-sm text-metal-text-muted">RNA 유형</span>
                      <p className="text-lg font-semibold text-metal-text-light">{result.sequence_analysis.detected_rna_type}</p>
                    </div>
                    <div>
                      <span className="text-sm text-metal-text-muted">유형 신뢰도</span>
                      <p className="text-lg font-semibold text-metal-text-light">
                        {(result.sequence_analysis.rna_type_confidence * 100).toFixed(0)}%
                      </p>
                    </div>
                  </div>

                  {result.sequence_analysis.motifs_found.length > 0 && (
                    <div className="mt-4">
                      <span className="text-sm text-metal-text-muted">발견된 모티프</span>
                      <div className="flex flex-wrap gap-2 mt-2">
                        {result.sequence_analysis.motifs_found.map((motif, idx) => (
                          <span key={idx} className="px-2 py-1 bg-accent-cyan/10 text-accent-cyan text-xs rounded-md">
                            {motif}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Risk Assessment */}
                <div className="bg-gradient-to-b from-[#2A2F37] to-[#252930] rounded-xl border border-white/10 p-6">
                  <h2 className="text-lg font-semibold text-metal-text-light mb-4">위험도 평가</h2>

                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <span className="text-sm text-metal-text-muted">위험 점수</span>
                      <p className={`text-3xl font-bold ${getRiskColor(result.risk_assessment.risk_level)}`}>
                        {result.risk_assessment.risk_score.toFixed(1)}
                      </p>
                    </div>
                    <div className={`px-4 py-2 rounded-lg ${getRiskColor(result.risk_assessment.risk_level)} bg-current/10`}>
                      <span className="font-semibold uppercase">{result.risk_assessment.risk_level}</span>
                    </div>
                  </div>

                  <div className="mb-4">
                    <div className="flex justify-between text-sm text-metal-text-muted mb-1">
                      <span>병원성</span>
                      <span>{result.risk_assessment.pathogenicity}</span>
                    </div>
                    <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${getRiskColor(result.risk_assessment.risk_level).replace('text-', 'bg-')}`}
                        style={{ width: `${result.risk_assessment.risk_score}%` }}
                      />
                    </div>
                  </div>

                  {result.risk_assessment.factors.length > 0 && (
                    <div className="mt-4">
                      <span className="text-sm text-metal-text-muted">위험 요인</span>
                      <ul className="mt-2 space-y-1">
                        {result.risk_assessment.factors.map((factor, idx) => (
                          <li key={idx} className="text-sm text-metal-text-mid flex items-center gap-2">
                            <span className="w-1.5 h-1.5 bg-yellow-400 rounded-full" />
                            {factor}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {result.risk_assessment.recommendations.length > 0 && (
                    <div className="mt-4">
                      <span className="text-sm text-metal-text-muted">권장 사항</span>
                      <ul className="mt-2 space-y-1">
                        {result.risk_assessment.recommendations.map((rec, idx) => (
                          <li key={idx} className="text-sm text-accent-cyan flex items-center gap-2">
                            <span className="w-1.5 h-1.5 bg-accent-cyan rounded-full" />
                            {rec}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>

                {/* Disease Predictions */}
                <div className="bg-gradient-to-b from-[#2A2F37] to-[#252930] rounded-xl border border-white/10 p-6">
                  <h2 className="text-lg font-semibold text-metal-text-light mb-4">질병 예측</h2>

                  <div className="space-y-4">
                    {result.disease_predictions.map((pred, idx) => (
                      <div key={idx} className="border-b border-white/5 pb-4 last:border-0 last:pb-0">
                        <div className="flex items-start justify-between">
                          <div>
                            <p className="font-medium text-metal-text-light">{pred.disease}</p>
                            <p className="text-sm text-metal-text-muted">{pred.disease_en}</p>
                            {pred.icd_code !== 'N/A' && (
                              <p className="text-xs text-metal-text-muted mt-1">ICD-10: {pred.icd_code}</p>
                            )}
                          </div>
                          <span className={`px-2 py-1 rounded-md text-xs font-medium border ${getConfidenceBadge(pred.confidence)}`}>
                            {pred.confidence.toUpperCase()}
                          </span>
                        </div>

                        <div className="mt-2">
                          <div className="flex justify-between text-sm text-metal-text-muted mb-1">
                            <span>확률</span>
                            <span>{(pred.probability * 100).toFixed(1)}%</span>
                          </div>
                          <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-accent-cyan transition-all"
                              style={{ width: `${pred.probability * 100}%` }}
                            />
                          </div>
                        </div>

                        {pred.related_genes.length > 0 && (
                          <div className="mt-2 flex flex-wrap gap-1">
                            {pred.related_genes.map((gene, gIdx) => (
                              <span key={gIdx} className="px-2 py-0.5 bg-blue-500/10 text-blue-400 text-xs rounded">
                                {gene}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Disclaimer */}
                <div className="p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                  <p className="text-sm text-yellow-400">{result.disclaimer}</p>
                </div>
              </>
            ) : (
              <div className="bg-gradient-to-b from-[#2A2F37] to-[#252930] rounded-xl border border-white/10 p-12 text-center">
                <div className="text-4xl mb-4">🧬</div>
                <h3 className="text-lg font-semibold text-metal-text-light mb-2">RNA 서열을 입력하세요</h3>
                <p className="text-sm text-metal-text-muted">
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
