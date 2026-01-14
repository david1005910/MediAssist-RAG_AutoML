import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { apiClient } from '../../services/api/client'

interface Experiment {
  experiment_id: string
  name: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped'
  n_trials_completed: number
  n_trials_total: number
  best_value?: number
  created_at: string
}

interface ExperimentConfig {
  experiment_name: string
  description?: string
  n_trials: number
  timeout_hours?: number
  sampler: 'tpe' | 'cma_es' | 'random'
  pruner: 'hyperband' | 'median' | 'none'
  objective_metric: string
  use_gpu: boolean
}

interface ExperimentPreset {
  name: string
  description: string
  icon: string
  config: ExperimentConfig
}

const EXPERIMENT_PRESETS: ExperimentPreset[] = [
  {
    name: '빠른 테스트',
    description: '5회 시도로 빠르게 기본 성능 확인',
    icon: '⚡',
    config: {
      experiment_name: 'quick_test',
      description: '빠른 성능 테스트 실험',
      n_trials: 5,
      timeout_hours: 1,
      sampler: 'random',
      pruner: 'none',
      objective_metric: 'f1_macro',
      use_gpu: true,
    },
  },
  {
    name: '표준 최적화',
    description: '50회 시도로 균형 잡힌 탐색',
    icon: '🎯',
    config: {
      experiment_name: 'standard_optimization',
      description: 'TPE 샘플러를 사용한 표준 하이퍼파라미터 최적화',
      n_trials: 50,
      timeout_hours: 6,
      sampler: 'tpe',
      pruner: 'hyperband',
      objective_metric: 'f1_macro',
      use_gpu: true,
    },
  },
  {
    name: '심층 탐색',
    description: '100회 이상 시도로 최적의 파라미터 탐색',
    icon: '🔬',
    config: {
      experiment_name: 'deep_exploration',
      description: 'CMA-ES를 사용한 심층 하이퍼파라미터 탐색',
      n_trials: 100,
      timeout_hours: 24,
      sampler: 'cma_es',
      pruner: 'hyperband',
      objective_metric: 'f1_macro',
      use_gpu: true,
    },
  },
  {
    name: 'RNA 질병 예측',
    description: 'RNA 서열 기반 질병 예측 모델 최적화',
    icon: '🧬',
    config: {
      experiment_name: 'rna_disease_prediction',
      description: 'RNA 서열 분석을 위한 Transformer 모델 최적화',
      n_trials: 30,
      timeout_hours: 12,
      sampler: 'tpe',
      pruner: 'hyperband',
      objective_metric: 'f1_macro',
      use_gpu: true,
    },
  },
]

export default function AutoMLDashboard() {
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null)
  const [config, setConfig] = useState<ExperimentConfig>({
    experiment_name: '',
    description: '',
    n_trials: 100,
    timeout_hours: 24,
    sampler: 'tpe',
    pruner: 'hyperband',
    objective_metric: 'f1_macro',
    use_gpu: true,
  })

  const applyPreset = (preset: ExperimentPreset) => {
    setConfig({
      ...preset.config,
      experiment_name: `${preset.config.experiment_name}_${Date.now().toString(36)}`,
    })
    setSelectedPreset(preset.name)
  }

  const { data: experiments = [], refetch, isLoading } = useQuery<Experiment[]>({
    queryKey: ['automl-experiments'],
    queryFn: async () => {
      return apiClient.listAutoMLExperiments()
    },
    refetchInterval: 5000,
  })

  const createMutation = useMutation({
    mutationFn: async (data: ExperimentConfig) => {
      return apiClient.createAutoMLExperiment(data)
    },
    onSuccess: () => {
      setShowCreateModal(false)
      setConfig({
        experiment_name: '',
        description: '',
        n_trials: 100,
        timeout_hours: 24,
        sampler: 'tpe',
        pruner: 'hyperband',
        objective_metric: 'f1_macro',
        use_gpu: true,
      })
      refetch()
    },
  })

  const stopMutation = useMutation({
    mutationFn: async (experimentId: string) => {
      return apiClient.stopAutoMLExperiment(experimentId)
    },
    onSuccess: () => refetch(),
  })

  const deleteMutation = useMutation({
    mutationFn: async (experimentId: string) => {
      return apiClient.deleteAutoMLExperiment(experimentId)
    },
    onSuccess: () => refetch(),
  })

  const getStatusBadge = (status: string) => {
    const styles: Record<string, string> = {
      running: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
      completed: 'bg-green-500/20 text-green-400 border-green-500/30',
      failed: 'bg-red-500/20 text-red-400 border-red-500/30',
      pending: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
      stopped: 'bg-gray-500/20 text-gray-400 border-gray-500/30',
    }
    return styles[status] || styles.pending
  }

  const runningCount = experiments.filter((e) => e.status === 'running').length
  const completedCount = experiments.filter((e) => e.status === 'completed').length
  const bestValue = experiments.length > 0
    ? Math.max(...experiments.filter((e) => e.best_value).map((e) => e.best_value || 0))
    : null

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="sticky top-0 z-50 metal-header px-4 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link to="/dashboard" className="text-white/70 hover:text-white transition-colors">
              ← Dashboard
            </Link>
            <h1 className="text-xl font-bold text-white">AutoML Experiments</h1>
          </div>
          <button
            onClick={() => setShowCreateModal(true)}
            className="metal-btn px-4 py-2"
          >
            + New Experiment
          </button>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <div className="metal-card p-4">
            <div className="text-metal-text-muted text-sm">Total Experiments</div>
            <div className="text-2xl font-bold text-metal-text-light">{experiments.length}</div>
          </div>
          <div className="metal-card p-4">
            <div className="text-metal-text-muted text-sm">Running</div>
            <div className="text-2xl font-bold text-blue-400">{runningCount}</div>
          </div>
          <div className="metal-card p-4">
            <div className="text-metal-text-muted text-sm">Completed</div>
            <div className="text-2xl font-bold text-green-400">{completedCount}</div>
          </div>
          <div className="metal-card p-4">
            <div className="text-metal-text-muted text-sm">Best F1 Score</div>
            <div className="text-2xl font-bold text-accent-cyan">
              {bestValue ? bestValue.toFixed(3) : '-'}
            </div>
          </div>
        </div>

        {/* Quick Start Guide */}
        {experiments.length === 0 && (
          <div className="metal-card p-6 mb-8">
            <h2 className="text-lg font-semibold text-metal-text-light mb-4">AutoML 시작 가이드</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="space-y-2">
                <div className="text-2xl">1️⃣</div>
                <h3 className="font-medium text-metal-text-light">프리셋 선택</h3>
                <p className="text-sm text-metal-text-muted">
                  "New Experiment" 버튼을 클릭하고 목적에 맞는 프리셋을 선택하세요.
                  빠른 테스트부터 심층 탐색까지 다양한 옵션이 있습니다.
                </p>
              </div>
              <div className="space-y-2">
                <div className="text-2xl">2️⃣</div>
                <h3 className="font-medium text-metal-text-light">실험 실행</h3>
                <p className="text-sm text-metal-text-muted">
                  실험이 시작되면 자동으로 하이퍼파라미터를 탐색합니다.
                  진행 상황을 실시간으로 확인할 수 있습니다.
                </p>
              </div>
              <div className="space-y-2">
                <div className="text-2xl">3️⃣</div>
                <h3 className="font-medium text-metal-text-light">결과 확인</h3>
                <p className="text-sm text-metal-text-muted">
                  완료된 실험에서 최적의 하이퍼파라미터와 성능 지표를 확인하세요.
                  앙상블 모델도 생성할 수 있습니다.
                </p>
              </div>
            </div>
            <div className="mt-6 p-4 bg-gray-100 rounded-lg">
              <h4 className="font-medium text-metal-text-light mb-2">하이퍼파라미터 탐색 범위</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-metal-text-muted">Learning Rate:</span>
                  <span className="text-metal-text-light ml-2">1e-6 ~ 1e-3</span>
                </div>
                <div>
                  <span className="text-metal-text-muted">Batch Size:</span>
                  <span className="text-metal-text-light ml-2">16, 32, 64</span>
                </div>
                <div>
                  <span className="text-metal-text-muted">Hidden Size:</span>
                  <span className="text-metal-text-light ml-2">256 ~ 768</span>
                </div>
                <div>
                  <span className="text-metal-text-muted">Num Layers:</span>
                  <span className="text-metal-text-light ml-2">4 ~ 12</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Experiments List */}
        <div className="metal-card">
          <div className="p-4 border-b border-white/10">
            <h2 className="text-lg font-semibold text-metal-text-light">Experiments</h2>
          </div>

          {isLoading ? (
            <div className="p-8 text-center text-metal-text-muted">Loading...</div>
          ) : experiments.length === 0 ? (
            <div className="p-8 text-center text-metal-text-muted">
              No experiments yet. Create your first AutoML experiment to get started.
            </div>
          ) : (
            <div className="divide-y divide-white/5">
              {experiments.map((experiment) => (
                <div
                  key={experiment.experiment_id}
                  className="p-4 hover:bg-gray-100 transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="font-medium text-metal-text-light">{experiment.name}</div>
                      <div className="text-sm text-metal-text-muted mt-1">
                        Created: {new Date(experiment.created_at).toLocaleString()}
                      </div>
                    </div>

                    <div className="flex items-center gap-4">
                      <div className="text-right">
                        <div className="text-sm text-metal-text-muted">
                          {experiment.n_trials_completed} / {experiment.n_trials_total} trials
                        </div>
                        {experiment.best_value && (
                          <div className="text-sm text-accent-cyan">
                            Best: {experiment.best_value.toFixed(4)}
                          </div>
                        )}
                      </div>

                      <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getStatusBadge(experiment.status)}`}>
                        {experiment.status.toUpperCase()}
                      </span>

                      <div className="flex gap-2">
                        {experiment.status === 'running' && (
                          <button
                            onClick={() => stopMutation.mutate(experiment.experiment_id)}
                            className="px-3 py-1 text-sm bg-yellow-500/20 text-yellow-400 hover:bg-yellow-500/30 rounded-lg transition-colors"
                          >
                            Stop
                          </button>
                        )}
                        <button
                          onClick={() => {
                            if (confirm('Are you sure you want to delete this experiment?')) {
                              deleteMutation.mutate(experiment.experiment_id)
                            }
                          }}
                          className="px-3 py-1 text-sm bg-red-500/20 text-red-400 hover:bg-red-500/30 rounded-lg transition-colors"
                        >
                          Delete
                        </button>
                      </div>
                    </div>
                  </div>

                  {experiment.status === 'running' && (
                    <div className="mt-3">
                      <div className="flex justify-between text-xs text-metal-text-muted mb-1">
                        <span>Progress</span>
                        <span>{((experiment.n_trials_completed / experiment.n_trials_total) * 100).toFixed(0)}%</span>
                      </div>
                      <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-accent-cyan transition-all"
                          style={{ width: `${(experiment.n_trials_completed / experiment.n_trials_total) * 100}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </main>

      {/* Create Experiment Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-[#2A2F37] border border-white/10 rounded-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
            <div className="p-4 border-b border-white/10 flex justify-between items-center">
              <h2 className="text-lg font-semibold text-metal-text-light">New AutoML Experiment</h2>
              <button
                onClick={() => setShowCreateModal(false)}
                className="text-metal-text-muted hover:text-metal-text-light"
              >
                X
              </button>
            </div>

            <div className="p-6 space-y-6">
              {/* Presets */}
              <div className="space-y-4">
                <h3 className="font-medium text-metal-text-light">실험 프리셋 선택</h3>
                <div className="grid grid-cols-2 gap-3">
                  {EXPERIMENT_PRESETS.map((preset) => (
                    <button
                      key={preset.name}
                      onClick={() => applyPreset(preset)}
                      className={`p-4 rounded-lg border text-left transition-all ${
                        selectedPreset === preset.name
                          ? 'border-accent-cyan bg-accent-cyan/10'
                          : 'border-gray-200 bg-white hover:border-gray-300'
                      }`}
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xl">{preset.icon}</span>
                        <span className="font-medium text-metal-text-light">{preset.name}</span>
                      </div>
                      <p className="text-xs text-metal-text-muted">{preset.description}</p>
                      <div className="mt-2 flex gap-2 text-xs">
                        <span className="px-2 py-0.5 bg-gray-100 rounded text-metal-text-mid">
                          {preset.config.n_trials}회
                        </span>
                        <span className="px-2 py-0.5 bg-gray-100 rounded text-metal-text-mid">
                          {preset.config.sampler.toUpperCase()}
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              <div className="border-t border-white/10 pt-4">
                <p className="text-xs text-metal-text-muted mb-4">
                  프리셋을 선택하거나 아래에서 직접 설정을 조정하세요.
                </p>
              </div>

              {/* Basic Info */}
              <div className="space-y-4">
                <h3 className="font-medium text-metal-text-light">기본 정보</h3>
                <div>
                  <label className="block text-sm text-metal-text-muted mb-1">실험 이름 *</label>
                  <input
                    type="text"
                    value={config.experiment_name}
                    onChange={(e) => setConfig({ ...config, experiment_name: e.target.value })}
                    className="w-full px-4 py-2 metal-input"
                    placeholder="예: rna_disease_v1"
                  />
                </div>
                <div>
                  <label className="block text-sm text-metal-text-muted mb-1">설명</label>
                  <textarea
                    value={config.description}
                    onChange={(e) => setConfig({ ...config, description: e.target.value })}
                    className="w-full px-4 py-2 metal-input h-20 resize-none"
                    placeholder="실험 목적을 설명하세요..."
                  />
                </div>
              </div>

              {/* Optimization Settings */}
              <div className="space-y-4">
                <h3 className="font-medium text-metal-text-light">최적화 설정</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm text-metal-text-muted mb-1">시도 횟수</label>
                    <input
                      type="number"
                      value={config.n_trials}
                      onChange={(e) => setConfig({ ...config, n_trials: parseInt(e.target.value) })}
                      className="w-full px-4 py-2 metal-input"
                      min={1}
                      max={10000}
                    />
                    <p className="text-xs text-metal-text-muted mt-1">권장: 빠른 테스트 5~10, 표준 50~100</p>
                  </div>
                  <div>
                    <label className="block text-sm text-metal-text-muted mb-1">제한 시간 (시간)</label>
                    <input
                      type="number"
                      value={config.timeout_hours}
                      onChange={(e) => setConfig({ ...config, timeout_hours: parseFloat(e.target.value) })}
                      className="w-full px-4 py-2 metal-input"
                      min={0.1}
                      max={720}
                      step={0.5}
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-metal-text-muted mb-1">샘플러</label>
                    <select
                      value={config.sampler}
                      onChange={(e) => setConfig({ ...config, sampler: e.target.value as any })}
                      className="w-full px-4 py-2 metal-input"
                    >
                      <option value="tpe">TPE (Tree-structured Parzen Estimator)</option>
                      <option value="cma_es">CMA-ES</option>
                      <option value="random">Random</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm text-metal-text-muted mb-1">프루너 (조기 종료)</label>
                    <select
                      value={config.pruner}
                      onChange={(e) => setConfig({ ...config, pruner: e.target.value as any })}
                      className="w-full px-4 py-2 metal-input"
                    >
                      <option value="hyperband">Hyperband (권장)</option>
                      <option value="median">Median</option>
                      <option value="none">없음</option>
                    </select>
                    <p className="text-xs text-metal-text-muted mt-1">성능이 낮은 시도를 조기에 중단</p>
                  </div>
                </div>
              </div>

              {/* Resource Settings */}
              <div className="space-y-4">
                <h3 className="font-medium text-metal-text-light">리소스 설정</h3>
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="use_gpu"
                    checked={config.use_gpu}
                    onChange={(e) => setConfig({ ...config, use_gpu: e.target.checked })}
                    className="rounded border-gray-300 bg-white"
                  />
                  <label htmlFor="use_gpu" className="text-sm text-metal-text-mid">GPU 가속 사용</label>
                </div>
              </div>

              {/* Info Box */}
              <div className="p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                <h4 className="font-medium text-blue-400 mb-2">실험 설명</h4>
                <ul className="text-sm text-metal-text-mid space-y-1">
                  <li>• <strong>TPE</strong>: Tree-structured Parzen Estimator - 효율적인 베이지안 최적화</li>
                  <li>• <strong>CMA-ES</strong>: 진화 전략 기반 - 연속적인 파라미터 탐색에 적합</li>
                  <li>• <strong>Random</strong>: 무작위 탐색 - 빠른 기본 테스트용</li>
                  <li>• <strong>Hyperband</strong>: 성능이 낮은 시도를 빠르게 제거하여 효율성 향상</li>
                </ul>
              </div>
            </div>

            <div className="p-4 border-t border-white/10 flex justify-end gap-3">
              <button
                onClick={() => setShowCreateModal(false)}
                className="px-4 py-2 text-metal-text-mid hover:text-metal-text-light transition-colors"
              >
                취소
              </button>
              <button
                onClick={() => createMutation.mutate(config)}
                disabled={!config.experiment_name || createMutation.isPending}
                className="px-4 py-2 bg-accent-cyan hover:bg-accent-cyan/80 disabled:bg-accent-cyan/30 disabled:cursor-not-allowed text-black font-semibold rounded-lg transition-colors"
              >
                {createMutation.isPending ? '생성 중...' : '실험 생성'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
