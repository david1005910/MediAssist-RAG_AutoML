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

export default function AutoMLDashboard() {
  const [showCreateModal, setShowCreateModal] = useState(false)
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
    <div className="min-h-screen bg-gradient-to-b from-[#1A1D21] to-[#15171A]">
      {/* Header */}
      <header className="sticky top-0 z-50 backdrop-blur-xl bg-[#1A1D21]/80 border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link to="/dashboard" className="text-metal-text-muted hover:text-metal-text-light transition-colors">
              ← Dashboard
            </Link>
            <h1 className="text-xl font-bold text-metal-text-light">AutoML Experiments</h1>
          </div>
          <button
            onClick={() => setShowCreateModal(true)}
            className="px-4 py-2 bg-accent-cyan hover:bg-accent-cyan/80 text-black font-semibold rounded-lg transition-colors"
          >
            + New Experiment
          </button>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-gradient-to-b from-[#2A2F37] to-[#252930] rounded-xl border border-white/10 p-4">
            <div className="text-metal-text-muted text-sm">Total Experiments</div>
            <div className="text-2xl font-bold text-metal-text-light">{experiments.length}</div>
          </div>
          <div className="bg-gradient-to-b from-[#2A2F37] to-[#252930] rounded-xl border border-white/10 p-4">
            <div className="text-metal-text-muted text-sm">Running</div>
            <div className="text-2xl font-bold text-blue-400">{runningCount}</div>
          </div>
          <div className="bg-gradient-to-b from-[#2A2F37] to-[#252930] rounded-xl border border-white/10 p-4">
            <div className="text-metal-text-muted text-sm">Completed</div>
            <div className="text-2xl font-bold text-green-400">{completedCount}</div>
          </div>
          <div className="bg-gradient-to-b from-[#2A2F37] to-[#252930] rounded-xl border border-white/10 p-4">
            <div className="text-metal-text-muted text-sm">Best F1 Score</div>
            <div className="text-2xl font-bold text-accent-cyan">
              {bestValue ? bestValue.toFixed(3) : '-'}
            </div>
          </div>
        </div>

        {/* Experiments List */}
        <div className="bg-gradient-to-b from-[#2A2F37] to-[#252930] rounded-xl border border-white/10">
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
                  className="p-4 hover:bg-white/5 transition-colors"
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
              {/* Basic Info */}
              <div className="space-y-4">
                <h3 className="font-medium text-metal-text-light">Basic Information</h3>
                <div>
                  <label className="block text-sm text-metal-text-muted mb-1">Experiment Name *</label>
                  <input
                    type="text"
                    value={config.experiment_name}
                    onChange={(e) => setConfig({ ...config, experiment_name: e.target.value })}
                    className="w-full px-4 py-2 bg-[#1A1D21] border border-white/10 rounded-lg text-metal-text-light focus:outline-none focus:ring-2 focus:ring-accent-cyan/50"
                    placeholder="e.g., rna_disease_v1"
                  />
                </div>
                <div>
                  <label className="block text-sm text-metal-text-muted mb-1">Description</label>
                  <textarea
                    value={config.description}
                    onChange={(e) => setConfig({ ...config, description: e.target.value })}
                    className="w-full px-4 py-2 bg-[#1A1D21] border border-white/10 rounded-lg text-metal-text-light focus:outline-none focus:ring-2 focus:ring-accent-cyan/50 h-20 resize-none"
                    placeholder="Describe the experiment objective..."
                  />
                </div>
              </div>

              {/* Optimization Settings */}
              <div className="space-y-4">
                <h3 className="font-medium text-metal-text-light">Optimization Settings</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm text-metal-text-muted mb-1">Number of Trials</label>
                    <input
                      type="number"
                      value={config.n_trials}
                      onChange={(e) => setConfig({ ...config, n_trials: parseInt(e.target.value) })}
                      className="w-full px-4 py-2 bg-[#1A1D21] border border-white/10 rounded-lg text-metal-text-light focus:outline-none focus:ring-2 focus:ring-accent-cyan/50"
                      min={1}
                      max={10000}
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-metal-text-muted mb-1">Timeout (hours)</label>
                    <input
                      type="number"
                      value={config.timeout_hours}
                      onChange={(e) => setConfig({ ...config, timeout_hours: parseFloat(e.target.value) })}
                      className="w-full px-4 py-2 bg-[#1A1D21] border border-white/10 rounded-lg text-metal-text-light focus:outline-none focus:ring-2 focus:ring-accent-cyan/50"
                      min={0.1}
                      max={720}
                      step={0.5}
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-metal-text-muted mb-1">Sampler</label>
                    <select
                      value={config.sampler}
                      onChange={(e) => setConfig({ ...config, sampler: e.target.value as any })}
                      className="w-full px-4 py-2 bg-[#1A1D21] border border-white/10 rounded-lg text-metal-text-light focus:outline-none focus:ring-2 focus:ring-accent-cyan/50"
                    >
                      <option value="tpe">TPE (Tree-structured Parzen Estimator)</option>
                      <option value="cma_es">CMA-ES</option>
                      <option value="random">Random</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm text-metal-text-muted mb-1">Pruner</label>
                    <select
                      value={config.pruner}
                      onChange={(e) => setConfig({ ...config, pruner: e.target.value as any })}
                      className="w-full px-4 py-2 bg-[#1A1D21] border border-white/10 rounded-lg text-metal-text-light focus:outline-none focus:ring-2 focus:ring-accent-cyan/50"
                    >
                      <option value="hyperband">Hyperband</option>
                      <option value="median">Median</option>
                      <option value="none">None</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Resource Settings */}
              <div className="space-y-4">
                <h3 className="font-medium text-metal-text-light">Resources</h3>
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="use_gpu"
                    checked={config.use_gpu}
                    onChange={(e) => setConfig({ ...config, use_gpu: e.target.checked })}
                    className="rounded border-white/10 bg-[#1A1D21]"
                  />
                  <label htmlFor="use_gpu" className="text-sm text-metal-text-mid">Use GPU acceleration</label>
                </div>
              </div>
            </div>

            <div className="p-4 border-t border-white/10 flex justify-end gap-3">
              <button
                onClick={() => setShowCreateModal(false)}
                className="px-4 py-2 text-metal-text-mid hover:text-metal-text-light transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => createMutation.mutate(config)}
                disabled={!config.experiment_name || createMutation.isPending}
                className="px-4 py-2 bg-accent-cyan hover:bg-accent-cyan/80 disabled:bg-accent-cyan/30 disabled:cursor-not-allowed text-black font-semibold rounded-lg transition-colors"
              >
                {createMutation.isPending ? 'Creating...' : 'Create Experiment'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
