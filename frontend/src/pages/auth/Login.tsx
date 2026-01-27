import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { useMutation } from '@tanstack/react-query'
import { apiClient } from '@/services/api/client'
import { supabaseAuth } from '@/services/supabase/client'
import { useAuthStore } from '@/stores/authStore'

const loginSchema = z.object({
  email: z.string().email('유효한 이메일을 입력하세요'),
  password: z.string().min(6, '비밀번호는 6자 이상이어야 합니다'),
})

type LoginFormData = z.infer<typeof loginSchema>

export default function Login() {
  const navigate = useNavigate()
  const login = useAuthStore((state) => state.login)
  const [error, setError] = useState<string | null>(null)
  const [useSupabase, setUseSupabase] = useState(false) // Default to MongoDB auth

  const {
    register,
    handleSubmit,
    setValue,
    formState: { errors },
  } = useForm<LoginFormData>({
    resolver: zodResolver(loginSchema),
  })

  // Supabase login mutation
  const supabaseMutation = useMutation({
    mutationFn: async (data: LoginFormData) => {
      const result = await supabaseAuth.signIn(data.email, data.password)
      return result
    },
    onSuccess: (data) => {
      if (data.user && data.session) {
        login(
          {
            id: data.user.id,
            email: data.user.email || '',
            name: data.user.user_metadata?.name || data.user.email?.split('@')[0] || 'User',
            role: data.user.user_metadata?.role || 'doctor',
          },
          data.session.access_token
        )
        localStorage.setItem('access_token', data.session.access_token)
        navigate('/dashboard')
      }
    },
    onError: (err: Error) => {
      if (err.message.includes('Invalid login credentials')) {
        setError('이메일 또는 비밀번호가 올바르지 않습니다.')
      } else if (err.message.includes('Email not confirmed')) {
        setError('이메일 인증이 필요합니다. 이메일을 확인해주세요.')
      } else {
        setError(err.message || '로그인 중 오류가 발생했습니다.')
      }
    },
  })

  // Local API login mutation (for demo)
  const localMutation = useMutation({
    mutationFn: (data: LoginFormData) => apiClient.login(data.email, data.password),
    onSuccess: (data) => {
      login(data.user, data.access_token)
      localStorage.setItem('access_token', data.access_token)
      navigate('/dashboard')
    },
    onError: () => {
      setError('이메일 또는 비밀번호가 올바르지 않습니다.')
    },
  })

  const mutation = useSupabase ? supabaseMutation : localMutation

  const onSubmit = (data: LoginFormData) => {
    setError(null)
    mutation.mutate(data)
  }

  const fillDemoCredentials = () => {
    setValue('email', 'demo@mediassist.ai')
    setValue('password', 'demo1234')
  }

  const loginWithDemo = () => {
    setError(null)
    setUseSupabase(false)
    localMutation.mutate({ email: 'demo@mediassist.ai', password: 'demo1234' })
  }

  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="max-w-md w-full space-y-8 p-8 metal-card">
        <div>
          <h2 className="text-center text-3xl font-bold text-gray-800">
            MediAssist AI
          </h2>
          <p className="mt-2 text-center text-sm text-gray-600">
            의료 진단 보조 시스템
          </p>
        </div>

        {/* Auth Method Toggle */}
        <div className="flex items-center justify-center gap-4 p-3 bg-gray-800 rounded-xl">
          <button
            type="button"
            onClick={() => setUseSupabase(true)}
            className={`px-4 py-2 text-sm font-medium rounded-lg transition-all ${
              useSupabase
                ? 'bg-indigo-600 text-white shadow-md'
                : 'bg-transparent text-gray-600 hover:bg-gray-700'
            }`}
          >
            Supabase 인증
          </button>
          <button
            type="button"
            onClick={() => setUseSupabase(false)}
            className={`px-4 py-2 text-sm font-medium rounded-lg transition-all ${
              !useSupabase
                ? 'bg-indigo-600 text-white shadow-md'
                : 'bg-transparent text-gray-600 hover:bg-gray-700'
            }`}
          >
            MongoDB 인증
          </button>
        </div>

        <form className="mt-6 space-y-6" onSubmit={handleSubmit(onSubmit)}>
          {error && (
            <div className="p-3 rounded-xl text-sm bg-red-900/30 border border-red-800 text-red-400">
              {error}
            </div>
          )}

          <div className="space-y-4">
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                이메일
              </label>
              <input
                {...register('email')}
                type="email"
                className="mt-1 block w-full px-4 py-3 metal-input"
                placeholder="your@email.com"
              />
              {errors.email && (
                <p className="mt-1 text-sm text-red-500">{errors.email.message}</p>
              )}
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-700">
                비밀번호
              </label>
              <input
                {...register('password')}
                type="password"
                className="mt-1 block w-full px-4 py-3 metal-input"
                placeholder="••••••••"
              />
              {errors.password && (
                <p className="mt-1 text-sm text-red-500">{errors.password.message}</p>
              )}
            </div>
          </div>

          <button
            type="submit"
            disabled={mutation.isPending}
            className="w-full flex justify-center py-3 px-4 text-sm font-medium metal-btn disabled:opacity-50"
          >
            {mutation.isPending ? '로그인 중...' : '로그인'}
          </button>
        </form>

        {/* Demo credentials */}
        <div className="mt-4 p-4 rounded-xl bg-indigo-900/30 border border-indigo-800">
          <p className="text-sm text-indigo-300 font-medium mb-2">데모 계정 (로컬 인증)</p>
          <p className="text-xs text-indigo-400">이메일: demo@mediassist.ai</p>
          <p className="text-xs text-indigo-400 mb-3">비밀번호: demo1234</p>
          <button
            type="button"
            onClick={loginWithDemo}
            disabled={localMutation.isPending}
            className="w-full text-sm py-2 px-4 metal-btn-secondary disabled:opacity-50"
          >
            {localMutation.isPending ? '로그인 중...' : '데모 계정으로 로그인'}
          </button>
        </div>

        {/* Register link */}
        <div className="text-center">
          <p className="text-sm text-gray-600">
            계정이 없으신가요?{' '}
            <Link to="/register" className="text-indigo-400 hover:text-indigo-300 font-medium transition-colors">
              회원가입
            </Link>
          </p>
        </div>
      </div>
    </div>
  )
}
