import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { useMutation } from '@tanstack/react-query'
import { apiClient } from '@/services/api/client'
import { supabaseAuth } from '@/services/supabase/client'
import { useAuthStore } from '@/stores/authStore'

const registerSchema = z.object({
  name: z.string().min(2, '이름은 2자 이상이어야 합니다'),
  email: z.string().email('유효한 이메일을 입력하세요'),
  password: z.string().min(6, '비밀번호는 6자 이상이어야 합니다'),
  confirmPassword: z.string(),
  role: z.enum(['doctor', 'nurse', 'researcher', 'admin']),
}).refine((data) => data.password === data.confirmPassword, {
  message: '비밀번호가 일치하지 않습니다',
  path: ['confirmPassword'],
})

type RegisterFormData = z.infer<typeof registerSchema>

const roleLabels = {
  doctor: '의사',
  nurse: '간호사',
  researcher: '연구원',
  admin: '관리자',
}

export default function Register() {
  const navigate = useNavigate()
  const login = useAuthStore((state) => state.login)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)
  const [useSupabase, setUseSupabase] = useState(false) // Default to MongoDB auth

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<RegisterFormData>({
    resolver: zodResolver(registerSchema),
    defaultValues: {
      role: 'doctor',
    },
  })

  // Supabase registration mutation
  const supabaseMutation = useMutation({
    mutationFn: async (data: RegisterFormData) => {
      const result = await supabaseAuth.signUp(data.email, data.password, {
        name: data.name,
        role: data.role,
      })
      return result
    },
    onSuccess: (data) => {
      if (data.user) {
        // Check if email confirmation is required
        if (data.user.identities?.length === 0) {
          setError('이 이메일은 이미 등록되어 있습니다.')
        } else if (!data.session) {
          // Email confirmation required
          setSuccess(true)
        } else {
          // Auto login if email confirmation not required
          navigate('/login')
        }
      }
    },
    onError: (err: Error) => {
      if (err.message.includes('already registered')) {
        setError('이미 등록된 이메일입니다.')
      } else if (err.message.includes('Password')) {
        setError('비밀번호가 요구사항을 충족하지 않습니다.')
      } else {
        setError(err.message || '회원가입에 실패했습니다.')
      }
    },
  })

  // MongoDB registration mutation
  const mongoMutation = useMutation({
    mutationFn: (data: RegisterFormData) =>
      apiClient.register({ email: data.email, password: data.password, name: data.name, role: data.role }),
    onSuccess: (data) => {
      login(data.user, data.access_token)
      localStorage.setItem('access_token', data.access_token)
      navigate('/dashboard')
    },
    onError: (err: Error) => {
      if (err.message.includes('이미 등록된')) {
        setError('이미 등록된 이메일입니다.')
      } else {
        setError(err.message || '회원가입에 실패했습니다.')
      }
    },
  })

  const mutation = useSupabase ? supabaseMutation : mongoMutation

  const onSubmit = (data: RegisterFormData) => {
    setError(null)
    setSuccess(false)
    mutation.mutate(data)
  }

  if (success) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="max-w-md w-full space-y-8 p-8 metal-card text-center">
          <div className="text-6xl mb-4">📧</div>
          <h2 className="text-2xl font-bold text-gray-800">이메일을 확인하세요</h2>
          <p className="text-gray-600 mt-4">
            회원가입 확인 이메일을 발송했습니다.
            <br />
            이메일의 링크를 클릭하여 가입을 완료하세요.
          </p>
          <Link
            to="/login"
            className="inline-block mt-6 px-6 py-3 metal-btn"
          >
            로그인 페이지로 이동
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex items-center justify-center py-12">
      <div className="max-w-md w-full space-y-8 p-8 metal-card">
        <div>
          <h2 className="text-center text-3xl font-bold text-gray-800">
            회원가입
          </h2>
          <p className="mt-2 text-center text-sm text-gray-600">
            MediAssist AI 계정을 만드세요
          </p>
        </div>

        {/* Auth Method Toggle */}
        <div className="flex items-center justify-center gap-4 p-3 bg-gray-100 rounded-xl">
          <button
            type="button"
            onClick={() => setUseSupabase(true)}
            className={`px-4 py-2 text-sm font-medium rounded-lg transition-all ${
              useSupabase
                ? 'bg-indigo-600 text-white shadow-md'
                : 'bg-transparent text-gray-600 hover:bg-gray-200'
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
                : 'bg-transparent text-gray-600 hover:bg-gray-200'
            }`}
          >
            MongoDB 인증
          </button>
        </div>

        <form className="mt-6 space-y-6" onSubmit={handleSubmit(onSubmit)}>
          {error && (
            <div className="p-3 rounded-xl text-sm bg-red-50 border border-red-200 text-red-600">
              {error}
            </div>
          )}

          <div className="space-y-4">
            {/* Name */}
            <div>
              <label htmlFor="name" className="block text-sm font-medium text-gray-700">
                이름
              </label>
              <input
                {...register('name')}
                type="text"
                className="mt-1 block w-full px-4 py-3 metal-input"
                placeholder="홍길동"
              />
              {errors.name && (
                <p className="mt-1 text-sm text-red-500">{errors.name.message}</p>
              )}
            </div>

            {/* Email */}
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                이메일
              </label>
              <input
                {...register('email')}
                type="email"
                className="mt-1 block w-full px-4 py-3 metal-input"
                placeholder="example@hospital.com"
              />
              {errors.email && (
                <p className="mt-1 text-sm text-red-500">{errors.email.message}</p>
              )}
            </div>

            {/* Role */}
            <div>
              <label htmlFor="role" className="block text-sm font-medium text-gray-700">
                직책
              </label>
              <select
                {...register('role')}
                className="mt-1 block w-full px-4 py-3 metal-select"
              >
                {Object.entries(roleLabels).map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
              {errors.role && (
                <p className="mt-1 text-sm text-red-500">{errors.role.message}</p>
              )}
            </div>

            {/* Password */}
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-700">
                비밀번호
              </label>
              <input
                {...register('password')}
                type="password"
                className="mt-1 block w-full px-4 py-3 metal-input"
                placeholder="6자 이상"
              />
              {errors.password && (
                <p className="mt-1 text-sm text-red-500">{errors.password.message}</p>
              )}
            </div>

            {/* Confirm Password */}
            <div>
              <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-700">
                비밀번호 확인
              </label>
              <input
                {...register('confirmPassword')}
                type="password"
                className="mt-1 block w-full px-4 py-3 metal-input"
                placeholder="비밀번호를 다시 입력하세요"
              />
              {errors.confirmPassword && (
                <p className="mt-1 text-sm text-red-500">{errors.confirmPassword.message}</p>
              )}
            </div>
          </div>

          <button
            type="submit"
            disabled={mutation.isPending}
            className="w-full flex justify-center py-3 px-4 text-sm font-medium metal-btn disabled:opacity-50"
          >
            {mutation.isPending ? '가입 중...' : '회원가입'}
          </button>
        </form>

        {/* Login link */}
        <div className="text-center">
          <p className="text-sm text-gray-600">
            이미 계정이 있으신가요?{' '}
            <Link to="/login" className="text-indigo-600 hover:text-indigo-800 font-medium">
              로그인
            </Link>
          </p>
        </div>
      </div>
    </div>
  )
}
