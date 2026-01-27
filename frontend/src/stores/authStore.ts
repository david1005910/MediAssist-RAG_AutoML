import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface User {
  id: string
  email: string
  name: string
  role: 'admin' | 'doctor' | 'nurse' | 'researcher'
}

interface AuthState {
  user: User | null
  accessToken: string | null
  isAuthenticated: boolean
  login: (user: User, token: string) => void
  logout: () => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      accessToken: null,
      isAuthenticated: false,
      login: (user, token) => {
        localStorage.setItem('access_token', token)
        set({
          user,
          accessToken: token,
          isAuthenticated: true,
        })
      },
      logout: () => {
        localStorage.removeItem('access_token')
        set({
          user: null,
          accessToken: null,
          isAuthenticated: false,
        })
      },
    }),
    {
      name: 'auth-storage',
    }
  )
)
