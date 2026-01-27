import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface Symptom {
  name: string
  severity: number
  duration_days: number
}

interface PatientInfo {
  age?: number
  gender?: 'male' | 'female'
  medical_history?: string[]
}

interface SymptomState {
  // 자연어 입력 텍스트
  freeText: string
  setFreeText: (text: string) => void

  // 추출된 증상 목록
  symptoms: Symptom[]
  setSymptoms: (symptoms: Symptom[]) => void

  // 환자 정보
  patientInfo: PatientInfo
  setPatientInfo: (info: PatientInfo) => void

  // 초기화
  clearAll: () => void
}

export const useSymptomStore = create<SymptomState>()(
  persist(
    (set) => ({
      freeText: '',
      setFreeText: (text) => set({ freeText: text }),

      symptoms: [{ name: '', severity: 5, duration_days: 1 }],
      setSymptoms: (symptoms) => set({ symptoms }),

      patientInfo: {},
      setPatientInfo: (info) => set({ patientInfo: info }),

      clearAll: () =>
        set({
          freeText: '',
          symptoms: [{ name: '', severity: 5, duration_days: 1 }],
          patientInfo: {},
        }),
    }),
    {
      name: 'symptom-analysis-storage',
    }
  )
)
