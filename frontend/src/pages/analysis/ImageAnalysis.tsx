import { useState, useRef, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { apiClient } from '@/services/api/client'
import type { ImageAnalysisResponse, GradCAMResponse } from '@/types'

export default function ImageAnalysis() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string>('')
  const [result, setResult] = useState<ImageAnalysisResponse | null>(null)
  const [gradcam, setGradcam] = useState<GradCAMResponse | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isGeneratingGradCAM, setIsGeneratingGradCAM] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // JSON viewer state
  const [selectedJsonData, setSelectedJsonData] = useState<object | null>(null)
  const [jsonCopied, setJsonCopied] = useState(false)

  // Cleanup object URL on unmount
  useEffect(() => {
    return () => {
      if (preview) {
        URL.revokeObjectURL(preview)
      }
    }
  }, [preview])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    if (preview) {
      URL.revokeObjectURL(preview)
    }

    setSelectedFile(file)
    setResult(null)
    setGradcam(null)
    setError(null)
    setPreview(URL.createObjectURL(file))
  }

  const handleAnalyze = async () => {
    if (!selectedFile || isAnalyzing) return

    setIsAnalyzing(true)
    setError(null)

    try {
      const data = await apiClient.analyzeImage(selectedFile)
      setResult(data)
      setGradcam(null)
    } catch (err) {
      console.error('Analysis error:', err)
      setError('분석 중 오류가 발생했습니다.')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleGradCAM = async (condition: string) => {
    if (!selectedFile || isGeneratingGradCAM) return

    setIsGeneratingGradCAM(true)

    try {
      const data = await apiClient.generateGradCAM(selectedFile, condition)
      setGradcam(data)
    } catch (err) {
      console.error('GradCAM error:', err)
    } finally {
      setIsGeneratingGradCAM(false)
    }
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

  const showAllResultsJson = () => {
    if (result) {
      setSelectedJsonData({
        analysis: result,
        gradcam: gradcam ? {
          condition: gradcam.condition,
          heatmap_base64: '[BASE64_IMAGE_DATA]',
          overlay_base64: '[BASE64_IMAGE_DATA]'
        } : null
      })
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#1A1D21] to-[#15171A]">
      <header className="metal-header">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center gap-4">
          <Link to="/dashboard" className="text-metal-text-muted hover:text-metal-text-light transition-colors">
            ← 대시보드
          </Link>
          <h1 className="text-xl font-bold text-metal-text-light">의료 이미지 분석</h1>
        </div>
      </header>

      <main className="max-w-full mx-auto px-4 py-8">
        <div className="flex gap-6">
          {/* Upload Section */}
          <div className="w-96 flex-shrink-0 space-y-6">
            <div className="metal-card p-6">
              <h2 className="text-lg font-semibold text-metal-text-light mb-4">흉부 X-ray 업로드</h2>

              <div
                onClick={() => fileInputRef.current?.click()}
                className="rounded-metal p-8 text-center cursor-pointer transition-all"
                style={{
                  background: 'linear-gradient(180deg, #252930 0%, #1F2328 100%)',
                  border: '2px dashed rgba(79, 195, 247, 0.3)',
                }}
              >
                {preview ? (
                  <img src={preview} alt="Preview" className="max-h-64 mx-auto rounded-metal" />
                ) : (
                  <div className="text-metal-text-muted">
                    <p className="text-4xl mb-2 opacity-50">📁</p>
                    <p>클릭하여 이미지 선택</p>
                    <p className="text-sm mt-1 text-metal-text-muted">PNG, JPG 지원</p>
                  </div>
                )}
              </div>

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
              />

              {selectedFile && (
                <div className="mt-4">
                  <p className="text-sm text-metal-text-muted mb-2">파일: {selectedFile.name}</p>
                  <button
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                    className="w-full py-3 px-4 metal-btn disabled:opacity-50"
                  >
                    {isAnalyzing ? '분석 중...' : '분석 시작'}
                  </button>
                  {error && (
                    <p className="mt-2 text-red-400 text-sm">{error}</p>
                  )}
                </div>
              )}
            </div>

            {result && (
              <div className="metal-card p-6">
                <h2 className="text-lg font-semibold text-metal-text-light mb-4">이미지 품질</h2>
                <div className="grid grid-cols-2 gap-4 text-sm text-metal-text-mid">
                  <div>해상도: {result.image_quality.resolution[0]} × {result.image_quality.resolution[1]}</div>
                  <div>밝기: {result.image_quality.brightness.toFixed(1)}</div>
                  <div>대비: {result.image_quality.contrast.toFixed(1)}</div>
                  <div>품질: <span className={result.image_quality.is_acceptable ? 'text-accent-green' : 'text-red-400'}>{result.image_quality.is_acceptable ? '적합' : '부적합'}</span></div>
                </div>
              </div>
            )}

            {/* Grad-CAM Section */}
            {gradcam && (
              <div className="metal-card p-6">
                <h2 className="text-lg font-semibold text-metal-text-light mb-4">Grad-CAM - {gradcam.condition}</h2>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-metal-text-muted mb-2">히트맵</p>
                    <img src={`data:image/png;base64,${gradcam.heatmap_base64}`} alt="Heatmap" className="rounded-metal border border-white/10" />
                  </div>
                  <div>
                    <p className="text-sm text-metal-text-muted mb-2">오버레이</p>
                    <img src={`data:image/png;base64,${gradcam.overlay_base64}`} alt="Overlay" className="rounded-metal border border-white/10" />
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Results Section */}
          <div className="flex-1 min-w-0 space-y-6">
            {isAnalyzing && (
              <div className="metal-card p-8 text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent-cyan mx-auto mb-4"></div>
                <p className="text-accent-cyan">이미지 분석 중...</p>
              </div>
            )}

            {result && (
              <>
                <div className="metal-card p-6">
                  <h2 className="text-lg font-semibold text-metal-text-light mb-4">분석 결과</h2>
                  <div className="space-y-3">
                    {result.findings.map((finding, index) => (
                      <div
                        key={index}
                        className="p-4 rounded-metal cursor-pointer transition-all hover:shadow-metal"
                        style={{
                          background: 'linear-gradient(180deg, #2A2F37 0%, #252930 100%)',
                          borderTop: '1px solid rgba(255,255,255,0.08)',
                          borderBottom: '1px solid rgba(0,0,0,0.3)'
                        }}
                        onClick={() => setSelectedJsonData(finding)}
                      >
                        <div className="flex justify-between items-center">
                          <div>
                            <h3 className="font-medium text-metal-text-light">{finding.condition}</h3>
                            <span className={`text-xs px-2 py-1 rounded-metal-sm ${getConfidenceColor(finding.confidence)}`}>
                              {finding.confidence === 'high' ? '높음' : finding.confidence === 'medium' ? '중간' : '낮음'}
                            </span>
                          </div>
                          <div className="text-xl font-bold text-accent-cyan">
                            {(finding.probability * 100).toFixed(1)}%
                          </div>
                        </div>
                        <div className="mt-2 h-2 metal-progress">
                          <div
                            className="h-full metal-progress-bar transition-all"
                            style={{ width: `${finding.probability * 100}%` }}
                          />
                        </div>
                        <button
                          onClick={(e) => { e.stopPropagation(); handleGradCAM(finding.condition); }}
                          disabled={isGeneratingGradCAM}
                          className="mt-2 text-sm text-accent-cyan hover:text-accent-cyan-light disabled:opacity-50 transition-colors"
                        >
                          {isGeneratingGradCAM ? 'Grad-CAM 생성 중...' : 'Grad-CAM 보기'}
                        </button>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="p-4 rounded-metal"
                  style={{
                    background: 'linear-gradient(180deg, #3D3A28 0%, #2E2B1F 100%)',
                    borderLeft: '3px solid #B7950B',
                    borderTop: '1px solid rgba(255,255,255,0.08)'
                  }}>
                  <p className="text-sm" style={{ color: '#D4AC0D' }}>
                    이 분석 결과는 참고용이며, 최종 진단은 반드시 전문의가 수행해야 합니다.
                  </p>
                </div>
              </>
            )}

            {!result && !isAnalyzing && (
              <div className="metal-card p-8 text-center text-metal-text-muted">
                흉부 X-ray 이미지를 업로드하고 분석을 시작하세요
              </div>
            )}
          </div>

          {/* Right Panel - JSON Viewer */}
          <div className="w-80 flex-shrink-0">
            <div className="metal-card sticky top-4">
              <div className="flex items-center justify-between p-4 border-b border-white/5">
                <h3 className="font-semibold text-metal-text-light text-sm">JSON 메타데이터</h3>
                <div className="flex gap-1">
                  <button
                    onClick={showAllResultsJson}
                    disabled={!result}
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
                    <p className="text-sm">분석 결과를 클릭하면</p>
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
