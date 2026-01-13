export interface User {
  id: string
  email: string
  name: string
  role: 'admin' | 'doctor' | 'nurse' | 'researcher'
}

// Symptom Analysis Types
export interface Symptom {
  name: string
  severity: number
  duration_days: number
}

export interface PatientInfo {
  age?: number
  gender?: 'male' | 'female'
  medical_history?: string[]
}

export interface DiseasePrediction {
  disease: string
  icd_code: string
  probability: number
  confidence: 'high' | 'medium' | 'low'
  description?: string
}

export interface RiskAssessment {
  risk_score: number
  risk_level: 'low' | 'medium' | 'high' | 'critical'
  factors: string[]
  recommendations: string[]
}

export interface SymptomAnalysisRequest {
  symptoms: Symptom[]
  patient_info?: PatientInfo
  top_k?: number
}

export interface SymptomAnalysisResponse {
  predictions: DiseasePrediction[]
  extracted_symptoms: string[]
  risk_assessment: RiskAssessment
  disclaimer: string
}

export interface SymptomNERRequest {
  text: string
}

export interface SymptomNERResponse {
  extracted_symptoms: Symptom[]
  original_text: string
  entities: Array<{ text: string; type: string }>
}

// Image Analysis Types
export interface ImageFinding {
  condition: string
  probability: number
  confidence: 'high' | 'medium' | 'low'
}

export interface ImageQuality {
  resolution: number[]
  brightness: number
  contrast: number
  is_acceptable: boolean
}

export interface ImageAnalysisResponse {
  findings: ImageFinding[]
  image_quality: ImageQuality
  gradcam_available: boolean
}

export interface GradCAMResponse {
  condition: string
  heatmap_base64: string
  overlay_base64: string
}

// RAG Types
export interface HybridScore {
  sparse_score: number
  dense_score: number
  combined_score: number
}

export interface DocumentSource {
  title: string
  authors?: string
  year?: string
  journal?: string
  pmid?: string
  relevance: number
  sparse_score?: number
  dense_score?: number
}

export interface SearchResult {
  content: string
  metadata: Record<string, string>
  score: number
  hybrid_score?: HybridScore
}

export interface LiteratureSearchRequest {
  query: string
  top_k?: number
}

export interface LiteratureSearchResponse {
  results: SearchResult[]
  total_found: number
  query: string
}

export interface RAGQueryRequest {
  question: string
  context?: {
    symptoms?: string[]
    patient_age?: number
  }
  include_sources?: boolean
}

export interface KnowledgeGraphNode {
  id: string
  label: string
  type: string
  properties: Record<string, unknown>
}

export interface KnowledgeGraphEdge {
  source: string
  target: string
  type: string
  properties: Record<string, unknown>
}

export interface KnowledgeGraphData {
  nodes: KnowledgeGraphNode[]
  edges: KnowledgeGraphEdge[]
  context: string
}

export interface RAGQueryResponse {
  answer: string
  sources: DocumentSource[]
  context_used?: string
  confidence: 'high' | 'medium' | 'low'
  knowledge_graph?: KnowledgeGraphData
  disclaimer: string
}

// Academic Search Types
export interface AcademicSource {
  id: string
  name: string
  description: string
}

export interface AcademicArticle {
  pmid: string
  title: string
  abstract: string
  authors: string
  year: string
  journal: string
  doi?: string
  keywords: string[]
  source: string
}

export interface AcademicSearchRequest {
  query: string
  sources?: string[]
  max_results_per_source?: number
  ingest?: boolean
}

export interface AcademicSearchResponse {
  query: string
  sources_searched: string[]
  results: Record<string, AcademicArticle[]>
  total_found: number
  ingested: number
}
