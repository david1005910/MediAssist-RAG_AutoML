import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { useMutation } from '@tanstack/react-query'
import { apiClient } from '@/services/api/client'
import { useAuthStore } from '@/stores/authStore'

const loginSchema = z.object({
  email: z.string().email('유효한 이메일을 입력하세요'),
  password: z.string().min(8, '비밀번호는 8자 이상이어야 합니다'),
})

type LoginFormData = z.infer<typeof loginSchema>

export default function Login() {
  const navigate = useNavigate()
  const login = useAuthStore((state) => state.login)
  const [error, setError] = useState<string | null>(null)

  const {
    register,
    handleSubmit,
    setValue,
    formState: { errors },
  } = useForm<LoginFormData>({
    resolver: zodResolver(loginSchema),
  })

  const mutation = useMutation({
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

  const onSubmit = (data: LoginFormData) => {
    setError(null)
    mutation.mutate(data)
  }

  const fillDemoCredentials = () => {
    setValue('email', 'demo@mediassist.ai')
    setValue('password', 'demo1234')
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-b from-[#1A1D21] to-[#15171A]">
      <div className="max-w-md w-full space-y-8 p-8 metal-card">
        <div>
          <h2 className="text-center text-3xl font-bold text-metal-text-light">
            MediAssist AI
          </h2>
          <p className="mt-2 text-center text-sm text-metal-text-muted">
            의료 진단 보조 시스템
          </p>
        </div>

        <form className="mt-8 space-y-6" onSubmit={handleSubmit(onSubmit)}>
          {error && (
            <div className="p-3 rounded-metal-sm text-sm"
              style={{
                background: 'linear-gradient(180deg, #5C2A2A 0%, #4A2222 100%)',
                borderTop: '1px solid rgba(255,255,255,0.1)',
                color: '#F0A0A0'
              }}>
              {error}
            </div>
          )}

          <div className="space-y-4">
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-metal-text-mid">
                이메일
              </label>
              <input
                {...register('email')}
                type="email"
                className="mt-1 block w-full px-4 py-3 metal-input"
                placeholder="your@email.com"
              />
              {errors.email && (
                <p className="mt-1 text-sm text-red-400">{errors.email.message}</p>
              )}
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-metal-text-mid">
                비밀번호
              </label>
              <input
                {...register('password')}
                type="password"
                className="mt-1 block w-full px-4 py-3 metal-input"
                placeholder="••••••••"
              />
              {errors.password && (
                <p className="mt-1 text-sm text-red-400">{errors.password.message}</p>
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
        <div className="mt-4 p-4 rounded-metal-sm"
          style={{
            background: 'linear-gradient(180deg, #2A3A4A 0%, #223344 100%)',
            borderTop: '1px solid rgba(79, 195, 247, 0.2)',
            borderBottom: '1px solid rgba(0,0,0,0.3)'
          }}>
          <p className="text-sm text-accent-cyan font-medium mb-2">데모 계정</p>
          <p className="text-xs text-metal-text-muted">이메일: demo@mediassist.ai</p>
          <p className="text-xs text-metal-text-muted mb-3">비밀번호: demo1234</p>
          <button
            type="button"
            onClick={fillDemoCredentials}
            className="w-full text-sm py-2 px-4 metal-btn-secondary"
          >
            데모 계정으로 로그인
          </button>
        </div>

        {/* Register link */}
        <div className="text-center">
          <p className="text-sm text-metal-text-muted">
            계정이 없으신가요?{' '}
            <Link to="/register" className="text-accent-cyan hover:text-accent-cyan-light font-medium transition-colors">
              회원가입
            </Link>
          </p>
        </div>
      </div>
    </div>
  )
}
