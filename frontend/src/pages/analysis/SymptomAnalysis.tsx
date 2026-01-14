import { useState, useEffect } from 'react'
import { useForm, useFieldArray } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { useMutation } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { apiClient } from '@/services/api/client'
import { useSymptomStore } from '@/stores/symptomStore'
import type { SymptomAnalysisResponse, SymptomNERResponse } from '@/types'

const symptomSchema = z.object({
  symptoms: z.array(z.object({
    name: z.string().min(1, '증상명을 입력하세요'),
    severity: z.number().min(1).max(10),
    duration_days: z.number().min(1),
  })).min(1, '최소 1개 이상의 증상을 입력하세요'),
  patient_info: z.object({
    age: z.number().optional(),
    gender: z.enum(['male', 'female']).optional(),
    medical_history: z.array(z.string()).optional(),
  }).optional(),
})

type SymptomFormData = z.infer<typeof symptomSchema>

export default function SymptomAnalysis() {
  const [result, setResult] = useState<SymptomAnalysisResponse | null>(null)
  const [extractedSymptoms, setExtractedSymptoms] = useState<SymptomNERResponse | null>(null)
  const [showCriteriaInfo, setShowCriteriaInfo] = useState(false)

  // 스토어에서 상태 가져오기 (페이지 이동 후에도 유지됨)
  const { freeText, setFreeText, symptoms: storedSymptoms, setSymptoms: setStoredSymptoms } = useSymptomStore()

  const {
    register,
    control,
    handleSubmit,
    formState: { errors },
    reset,
  } = useForm<SymptomFormData>({
    resolver: zodResolver(symptomSchema),
    defaultValues: {
      symptoms: storedSymptoms.length > 0 && storedSymptoms[0].name
        ? storedSymptoms
        : [{ name: '', severity: 5, duration_days: 1 }],
      patient_info: { age: undefined, gender: undefined, medical_history: [] },
    },
  })

  const { fields, append, remove, replace } = useFieldArray({
    control,
    name: 'symptoms',
  })

  // 저장된 증상이 있으면 폼에 반영
  useEffect(() => {
    if (storedSymptoms.length > 0 && storedSymptoms[0].name) {
      reset({
        symptoms: storedSymptoms,
        patient_info: { age: undefined, gender: undefined, medical_history: [] },
      })
    }
  }, [])

  const analysisMutation = useMutation({
    mutationFn: (data: SymptomFormData) => {
      console.log('Sending analysis request:', data)
      return apiClient.analyzeSymptoms({
        symptoms: data.symptoms,
        patient_info: data.patient_info,
        top_k: 5,
      })
    },
    onSuccess: (data) => {
      console.log('Analysis result:', data)
      setResult(data)
    },
    onError: (error) => {
      console.error('Analysis error:', error)
    },
  })

  const extractMutation = useMutation({
    mutationFn: (text: string) => {
      console.log('Extracting symptoms from:', text)
      return apiClient.extractSymptoms({ text })
    },
    onSuccess: (data) => {
      console.log('Extraction result:', data)
      setExtractedSymptoms(data)
      // Auto-fill symptoms from extraction using replace
      if (data.extracted_symptoms.length > 0) {
        const newSymptoms = data.extracted_symptoms.map((symptom) => ({
          name: symptom.name,
          severity: symptom.severity,
          duration_days: symptom.duration_days,
        }))
        replace(newSymptoms)
        // 스토어에도 저장 (페이지 이동 시 유지)
        setStoredSymptoms(newSymptoms)
      }
    },
    onError: (error) => {
      console.error('Extraction error:', error)
    },
  })

  const onSubmit = (data: SymptomFormData) => {
    console.log('Form submitted with data:', data)
    // Filter out empty symptom names
    const validSymptoms = data.symptoms.filter(s => s.name.trim() !== '')
    if (validSymptoms.length === 0) {
      alert('최소 1개 이상의 증상을 입력하세요')
      return
    }
    analysisMutation.mutate({
      ...data,
      symptoms: validSymptoms,
    })
  }

  const handleExtractSymptoms = () => {
    if (freeText.trim()) {
      extractMutation.mutate(freeText)
    }
  }

  const getRiskLevelStyle = (level: string) => {
    switch (level) {
      case 'critical': return {
        bg: 'linear-gradient(180deg, #5C2A2A 0%, #4A2222 100%)',
        border: '#8B3A3A',
        text: '#F0A0A0'
      }
      case 'high': return {
        bg: 'linear-gradient(180deg, #5C3A2A 0%, #4A2E22 100%)',
        border: '#E67E22',
        text: '#F5B041'
      }
      case 'medium': return {
        bg: 'linear-gradient(180deg, #3D3A28 0%, #2E2B1F 100%)',
        border: '#B7950B',
        text: '#D4AC0D'
      }
      default: return {
        bg: 'linear-gradient(180deg, #2A3D2A 0%, #1F2E1F 100%)',
        border: '#26C281',
        text: '#2ECC71'
      }
    }
  }

  const getConfidenceColor = (confidence: string) => {
    switch (confidence) {
      case 'high': return 'text-accent-green'
      case 'medium': return 'text-yellow-500'
      default: return 'text-metal-text-muted'
    }
  }

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="metal-header px-4 py-4">
        <div className="max-w-7xl mx-auto flex items-center gap-4">
          <Link to="/dashboard" className="text-white/70 hover:text-white transition-colors">
            ← 대시보드
          </Link>
          <h1 className="text-xl font-bold text-white">증상 분석</h1>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-8">
        {/* Criteria Information Panel */}
        <div className="mb-6">
          <button
            onClick={() => setShowCriteriaInfo(!showCriteriaInfo)}
            className="w-full flex items-center justify-between px-4 py-3 rounded-metal transition-all"
            style={{
              background: 'linear-gradient(180deg, #2A3A4A 0%, #223344 100%)',
              borderTop: '1px solid rgba(79, 195, 247, 0.2)',
              borderBottom: '1px solid rgba(0,0,0,0.3)'
            }}
          >
            <span className="font-medium text-accent-cyan">위험도 평가 기준 및 유사도 검색 시스템 안내</span>
            <span className="text-accent-cyan">{showCriteriaInfo ? '▲ 닫기' : '▼ 펼치기'}</span>
          </button>

          {showCriteriaInfo && (
            <div className="mt-4 metal-card p-6 space-y-6">
              {/* Risk Score Calculation */}
              <div>
                <h3 className="text-lg font-semibold text-metal-text-light mb-3 flex items-center gap-2">
                  <span className="text-xl">📊</span> 위험도 점수 계산 요소
                </h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm border-collapse">
                    <thead>
                      <tr style={{ background: 'linear-gradient(180deg, #2C3036 0%, #23272B 100%)' }}>
                        <th className="border border-gray-200 px-3 py-2 text-left text-metal-text-light">요소</th>
                        <th className="border border-gray-200 px-3 py-2 text-left text-metal-text-light">조건</th>
                        <th className="border border-gray-200 px-3 py-2 text-left text-metal-text-light">점수</th>
                      </tr>
                    </thead>
                    <tbody className="text-metal-text-mid">
                      <tr style={{ background: 'rgba(0,0,0,0.2)' }}>
                        <td className="border border-gray-200 px-3 py-2 font-medium">기본 위험도</td>
                        <td className="border border-gray-200 px-3 py-2">질병 위험도 가중치 x 점수 x 100</td>
                        <td className="border border-gray-200 px-3 py-2">계산값</td>
                      </tr>
                      <tr>
                        <td className="border border-gray-200 px-3 py-2 font-medium">증상 심각도</td>
                        <td className="border border-gray-200 px-3 py-2">최대 심각도 7 이상</td>
                        <td className="border border-gray-200 px-3 py-2 text-orange-400 font-medium">+15점</td>
                      </tr>
                      <tr style={{ background: 'rgba(0,0,0,0.2)' }}>
                        <td className="border border-gray-200 px-3 py-2 font-medium">지속 기간</td>
                        <td className="border border-gray-200 px-3 py-2">7일 이상 지속</td>
                        <td className="border border-gray-200 px-3 py-2 text-orange-400 font-medium">+10점</td>
                      </tr>
                      <tr>
                        <td className="border border-gray-200 px-3 py-2 font-medium" rowSpan={2}>나이</td>
                        <td className="border border-gray-200 px-3 py-2">65세 이상</td>
                        <td className="border border-gray-200 px-3 py-2 text-red-400 font-medium">+15점</td>
                      </tr>
                      <tr>
                        <td className="border border-gray-200 px-3 py-2">5세 이하</td>
                        <td className="border border-gray-200 px-3 py-2 text-orange-400 font-medium">+10점</td>
                      </tr>
                      <tr style={{ background: 'rgba(0,0,0,0.2)' }}>
                        <td className="border border-gray-200 px-3 py-2 font-medium">기저 질환</td>
                        <td className="border border-gray-200 px-3 py-2">과거 병력 각각</td>
                        <td className="border border-gray-200 px-3 py-2 text-yellow-400 font-medium">+5점</td>
                      </tr>
                      <tr>
                        <td className="border border-gray-200 px-3 py-2 font-medium">다중 증상</td>
                        <td className="border border-gray-200 px-3 py-2">5개 이상 증상</td>
                        <td className="border border-gray-200 px-3 py-2 text-yellow-400 font-medium">+5점</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Risk Level Criteria */}
              <div>
                <h3 className="text-lg font-semibold text-metal-text-light mb-3 flex items-center gap-2">
                  <span className="text-xl">⚠️</span> 위험도 레벨 분류
                </h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm border-collapse">
                    <thead>
                      <tr style={{ background: 'linear-gradient(180deg, #2C3036 0%, #23272B 100%)' }}>
                        <th className="border border-gray-200 px-3 py-2 text-left text-metal-text-light">레벨</th>
                        <th className="border border-gray-200 px-3 py-2 text-left text-metal-text-light">점수 범위</th>
                        <th className="border border-gray-200 px-3 py-2 text-left text-metal-text-light">의미</th>
                      </tr>
                    </thead>
                    <tbody className="text-metal-text-mid">
                      <tr style={{ background: 'rgba(0,0,0,0.2)' }}>
                        <td className="border border-gray-200 px-3 py-2">
                          <span className="px-2 py-1 metal-badge-red rounded font-medium">CRITICAL</span>
                        </td>
                        <td className="border border-gray-200 px-3 py-2 font-medium">70점 이상</td>
                        <td className="border border-gray-200 px-3 py-2">즉시 의료 조치 권장</td>
                      </tr>
                      <tr>
                        <td className="border border-gray-200 px-3 py-2">
                          <span className="px-2 py-1 rounded font-medium" style={{ background: 'linear-gradient(180deg, #E67E22 0%, #D35400 100%)', color: '#FFF' }}>HIGH</span>
                        </td>
                        <td className="border border-gray-200 px-3 py-2 font-medium">50 ~ 69점</td>
                        <td className="border border-gray-200 px-3 py-2">24시간 내 진료 권장</td>
                      </tr>
                      <tr style={{ background: 'rgba(0,0,0,0.2)' }}>
                        <td className="border border-gray-200 px-3 py-2">
                          <span className="px-2 py-1 metal-badge-yellow rounded font-medium">MEDIUM</span>
                        </td>
                        <td className="border border-gray-200 px-3 py-2 font-medium">30 ~ 49점</td>
                        <td className="border border-gray-200 px-3 py-2">가까운 시일 내 진료 권장</td>
                      </tr>
                      <tr>
                        <td className="border border-gray-200 px-3 py-2">
                          <span className="px-2 py-1 metal-badge-green rounded font-medium">LOW</span>
                        </td>
                        <td className="border border-gray-200 px-3 py-2 font-medium">30점 미만</td>
                        <td className="border border-gray-200 px-3 py-2">자가 관리 가능, 필요시 진료</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Similarity Search System */}
              <div>
                <h3 className="text-lg font-semibold text-metal-text-light mb-3 flex items-center gap-2">
                  <span className="text-xl">🔍</span> Qdrant 하이브리드 유사도 검색 시스템
                </h3>
                <div className="rounded-metal p-4 space-y-3"
                  style={{
                    background: 'linear-gradient(180deg, #2A2F37 0%, #252930 100%)',
                    border: '1px solid rgba(155, 89, 182, 0.3)'
                  }}>
                  <div className="flex gap-2 mb-2">
                    <span className="px-2 py-1 rounded text-xs font-medium" style={{ background: 'linear-gradient(180deg, #E67E22 0%, #D35400 100%)', color: '#FFF' }}>Sparse 30%</span>
                    <span className="px-2 py-1 metal-badge rounded text-xs font-medium">Dense 70%</span>
                  </div>
                  <div>
                    <h4 className="font-medium text-orange-400 mb-1">1. Sparse Search (SPLADE/BM42)</h4>
                    <p className="text-sm text-metal-text-mid">
                      키워드 기반 희소 벡터 검색입니다. 정확한 용어 매칭에 강점이 있습니다.
                      <br />
                      <code className="bg-black/30 px-1 rounded text-xs text-metal-text-muted">가중치: 30%</code>
                    </p>
                  </div>
                  <div>
                    <h4 className="font-medium text-accent-cyan mb-1">2. Dense Search (BioBERT)</h4>
                    <p className="text-sm text-metal-text-mid">
                      의학 전문 언어 모델을 사용한 의미 기반 밀집 벡터 검색입니다.
                      문맥과 개념 이해에 강점이 있습니다.
                      <br />
                      <code className="bg-black/30 px-1 rounded text-xs text-metal-text-muted">가중치: 70%</code>
                    </p>
                  </div>
                  <div>
                    <h4 className="font-medium mb-1" style={{ color: '#B39DDB' }}>3. QdrantDB 하이브리드 검색</h4>
                    <p className="text-sm text-metal-text-mid">
                      Qdrant 벡터 데이터베이스로 Sparse + Dense 검색을 동시에 수행합니다.
                      각 검색 결과를 정규화하여 가중 합산합니다.
                    </p>
                  </div>
                  <div>
                    <h4 className="font-medium text-accent-green mb-1">4. 최종 점수 계산</h4>
                    <p className="text-sm text-metal-text-mid">
                      <code className="bg-black/30 px-1 rounded text-xs text-metal-text-muted">종합 점수 = (Sparse × 0.3) + (Dense × 0.7)</code>
                      <br />
                      신뢰도는 확률에 따라 high(70%↑), medium(40-70%), low(40%↓)로 분류됩니다.
                    </p>
                  </div>
                </div>
              </div>

              <div className="text-xs text-metal-text-muted pt-2 border-t border-gray-200">
                * 이 시스템은 참고용이며, 최종 진단은 반드시 의료 전문가와 상담하세요.
              </div>
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Section */}
          <div className="space-y-6">
            {/* Free Text Input */}
            <div className="metal-card p-6">
              <h2 className="text-lg font-semibold text-metal-text-light mb-4">자연어 증상 입력</h2>
              <textarea
                value={freeText}
                onChange={(e) => setFreeText(e.target.value)}
                placeholder="예: 3일 전부터 열이 나고 기침이 심해졌습니다. 머리도 아프고 몸살 기운이 있어요."
                className="w-full px-3 py-2 metal-input h-24 resize-none"
              />
              <button
                type="button"
                onClick={handleExtractSymptoms}
                disabled={extractMutation.isPending || !freeText.trim()}
                className="mt-2 px-4 py-2 metal-btn disabled:opacity-50"
              >
                {extractMutation.isPending ? '추출 중...' : '증상 추출'}
              </button>
              {extractedSymptoms && (
                <p className="mt-2 text-sm text-accent-green">
                  {extractedSymptoms.extracted_symptoms.length}개 증상이 추출되었습니다.
                </p>
              )}
            </div>

            {/* Manual Symptom Input */}
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
              <div className="metal-card p-6">
                <h2 className="text-lg font-semibold text-metal-text-light mb-4">증상 목록</h2>

                {fields.map((field, index) => (
                  <div key={field.id} className="flex gap-2 mb-3 items-start">
                    <div className="flex-1">
                      <input
                        {...register(`symptoms.${index}.name`)}
                        placeholder="증상명"
                        className="w-full px-3 py-2 metal-input text-sm"
                      />
                      {errors.symptoms?.[index]?.name && (
                        <p className="mt-1 text-xs text-red-400">
                          {errors.symptoms[index]?.name?.message}
                        </p>
                      )}
                    </div>
                    <div className="w-20">
                      <input
                        type="number"
                        {...register(`symptoms.${index}.severity`, { valueAsNumber: true })}
                        placeholder="심각도"
                        min={1}
                        max={10}
                        className="w-full px-2 py-2 metal-input text-sm"
                      />
                      <span className="text-xs text-metal-text-muted">1-10</span>
                    </div>
                    <div className="w-20">
                      <input
                        type="number"
                        {...register(`symptoms.${index}.duration_days`, { valueAsNumber: true })}
                        placeholder="일수"
                        min={1}
                        className="w-full px-2 py-2 metal-input text-sm"
                      />
                      <span className="text-xs text-metal-text-muted">일</span>
                    </div>
                    <button
                      type="button"
                      onClick={() => remove(index)}
                      className="px-2 py-2 text-red-400 hover:text-red-300 transition-colors"
                    >
                      ✕
                    </button>
                  </div>
                ))}

                <button
                  type="button"
                  onClick={() => append({ name: '', severity: 5, duration_days: 1 })}
                  className="text-accent-cyan hover:text-accent-cyan-light text-sm transition-colors"
                >
                  + 증상 추가
                </button>
              </div>

              {/* Patient Info */}
              <div className="metal-card p-6">
                <h2 className="text-lg font-semibold text-metal-text-light mb-4">환자 정보 (선택)</h2>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm text-metal-text-mid mb-1">나이</label>
                    <input
                      type="number"
                      {...register('patient_info.age', { valueAsNumber: true })}
                      placeholder="나이"
                      className="w-full px-3 py-2 metal-input"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-metal-text-mid mb-1">성별</label>
                    <select
                      {...register('patient_info.gender')}
                      className="w-full px-3 py-2 metal-select"
                    >
                      <option value="">선택</option>
                      <option value="male">남성</option>
                      <option value="female">여성</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Submit */}
              <button
                type="submit"
                disabled={analysisMutation.isPending}
                className="w-full py-3 px-4 metal-btn disabled:opacity-50 font-medium"
              >
                {analysisMutation.isPending ? '분석 중...' : '증상 분석'}
              </button>

              {errors.symptoms && (
                <p className="mt-2 text-sm text-red-400 text-center">
                  {errors.symptoms.message || '증상을 입력해주세요'}
                </p>
              )}
            </form>
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            {analysisMutation.isError && (
              <div className="p-4 rounded-metal"
                style={{
                  background: 'linear-gradient(180deg, #5C2A2A 0%, #4A2222 100%)',
                  borderTop: '1px solid rgba(255,255,255,0.1)',
                  borderLeft: '3px solid #8B3A3A'
                }}>
                <p className="font-medium text-red-300">분석 요청 중 오류가 발생했습니다.</p>
                <p className="text-sm mt-1 text-red-400">
                  {(analysisMutation.error as any)?.response?.data?.detail ||
                   (analysisMutation.error as Error)?.message ||
                   '다시 시도해주세요.'}
                </p>
              </div>
            )}

            {extractMutation.isError && (
              <div className="p-4 rounded-metal"
                style={{
                  background: 'linear-gradient(180deg, #5C2A2A 0%, #4A2222 100%)',
                  borderTop: '1px solid rgba(255,255,255,0.1)',
                  borderLeft: '3px solid #8B3A3A'
                }}>
                <p className="font-medium text-red-300">증상 추출 중 오류가 발생했습니다.</p>
                <p className="text-sm mt-1 text-red-400">
                  {(extractMutation.error as any)?.response?.data?.detail ||
                   (extractMutation.error as Error)?.message ||
                   '다시 시도해주세요.'}
                </p>
              </div>
            )}

            {result && (
              <>
                {/* Risk Assessment */}
                <div className="rounded-metal p-6"
                  style={{
                    background: getRiskLevelStyle(result.risk_assessment.risk_level).bg,
                    borderLeft: `3px solid ${getRiskLevelStyle(result.risk_assessment.risk_level).border}`,
                    borderTop: '1px solid rgba(255,255,255,0.08)'
                  }}>
                  <div className="flex justify-between items-center mb-4">
                    <h2 className="text-lg font-semibold text-metal-text-light">위험도 평가</h2>
                    <span className="text-2xl font-bold" style={{ color: getRiskLevelStyle(result.risk_assessment.risk_level).text }}>
                      {result.risk_assessment.risk_score.toFixed(0)}점
                    </span>
                  </div>
                  <div className="mb-4">
                    <span className="px-3 py-1 rounded-metal-sm text-sm font-medium uppercase"
                      style={{
                        background: getRiskLevelStyle(result.risk_assessment.risk_level).border,
                        color: '#FFF'
                      }}>
                      {result.risk_assessment.risk_level}
                    </span>
                  </div>
                  {result.risk_assessment.factors.length > 0 && (
                    <div className="mb-4">
                      <h3 className="text-sm font-medium mb-2 text-metal-text-light">위험 요인</h3>
                      <ul className="text-sm space-y-1" style={{ color: getRiskLevelStyle(result.risk_assessment.risk_level).text }}>
                        {result.risk_assessment.factors.map((factor, i) => (
                          <li key={i}>• {factor}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {result.risk_assessment.recommendations.length > 0 && (
                    <div>
                      <h3 className="text-sm font-medium mb-2 text-metal-text-light">권장 사항</h3>
                      <ul className="text-sm space-y-1" style={{ color: getRiskLevelStyle(result.risk_assessment.risk_level).text }}>
                        {result.risk_assessment.recommendations.map((rec, i) => (
                          <li key={i}>• {rec}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>

                {/* Disease Predictions */}
                <div className="metal-card p-6">
                  <h2 className="text-lg font-semibold text-metal-text-light mb-4">예측 질환</h2>
                  <div className="space-y-3">
                    {result.predictions.map((pred, index) => (
                      <div key={index} className="rounded-metal p-4"
                        style={{
                          background: 'linear-gradient(180deg, #2A2F37 0%, #252930 100%)',
                          borderTop: '1px solid rgba(255,255,255,0.08)',
                          borderBottom: '1px solid rgba(0,0,0,0.3)'
                        }}>
                        <div className="flex justify-between items-start">
                          <div>
                            <h3 className="font-medium text-metal-text-light">{pred.disease}</h3>
                            <p className="text-sm text-metal-text-muted">ICD-10: {pred.icd_code}</p>
                            {pred.description && (
                              <p className="text-sm text-metal-text-mid mt-1">{pred.description}</p>
                            )}
                          </div>
                          <div className="text-right">
                            <div className="text-lg font-semibold text-accent-cyan">
                              {(pred.probability * 100).toFixed(1)}%
                            </div>
                            <span className={`text-sm ${getConfidenceColor(pred.confidence)}`}>
                              {pred.confidence === 'high' ? '높음' : pred.confidence === 'medium' ? '중간' : '낮음'}
                            </span>
                          </div>
                        </div>
                        {/* Progress bar */}
                        <div className="mt-2 h-2 metal-progress">
                          <div
                            className="h-full metal-progress-bar transition-all"
                            style={{ width: `${pred.probability * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Extracted Symptoms */}
                <div className="metal-card p-6">
                  <h2 className="text-lg font-semibold text-metal-text-light mb-4">분석된 증상</h2>
                  <div className="flex flex-wrap gap-2">
                    {result.extracted_symptoms.map((symptom, index) => (
                      <span
                        key={index}
                        className="px-3 py-1 metal-badge rounded-metal-sm text-sm"
                      >
                        {symptom}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Disclaimer */}
                <div className="p-4 rounded-metal"
                  style={{
                    background: 'linear-gradient(180deg, #3D3A28 0%, #2E2B1F 100%)',
                    borderLeft: '3px solid #B7950B',
                    borderTop: '1px solid rgba(255,255,255,0.08)'
                  }}>
                  <p className="text-sm" style={{ color: '#D4AC0D' }}>{result.disclaimer}</p>
                </div>
              </>
            )}

            {!result && !analysisMutation.isPending && (
              <div className="metal-card p-8 text-center text-metal-text-muted">
                증상을 입력하고 분석을 시작하세요
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
