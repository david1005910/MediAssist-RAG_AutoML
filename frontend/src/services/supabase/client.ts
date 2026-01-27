import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'https://eflsbnsivzjdrhflvtpb.supabase.co'
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inludm9qeHNxdGhqcm1jcXZ4ZWhqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQ3NjAxMjgsImV4cCI6MjA4MDMzNjEyOH0.X2v8H4uhPBsy5_404MUVK7MvMtR78mkBpP8j3U_BnIY'

export const supabase = createClient(supabaseUrl, supabaseAnonKey)

// Auth helper functions
export const supabaseAuth = {
  // Sign up with email and password
  signUp: async (email: string, password: string, metadata?: { name?: string; role?: string }) => {
    const { data, error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        data: metadata,
      },
    })
    if (error) throw error
    return data
  },

  // Sign in with email and password
  signIn: async (email: string, password: string) => {
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password,
    })
    if (error) throw error
    return data
  },

  // Sign out
  signOut: async () => {
    const { error } = await supabase.auth.signOut()
    if (error) throw error
  },

  // Get current user
  getCurrentUser: async () => {
    const { data: { user }, error } = await supabase.auth.getUser()
    if (error) throw error
    return user
  },

  // Get session
  getSession: async () => {
    const { data: { session }, error } = await supabase.auth.getSession()
    if (error) throw error
    return session
  },

  // Listen to auth state changes
  onAuthStateChange: (callback: (event: string, session: unknown) => void) => {
    return supabase.auth.onAuthStateChange(callback)
  },

  // Reset password
  resetPassword: async (email: string) => {
    const { data, error } = await supabase.auth.resetPasswordForEmail(email)
    if (error) throw error
    return data
  },
}
