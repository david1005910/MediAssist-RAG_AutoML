# IMPLEMENTATION.md
# MediAssist AI - 구현 가이드

---

## 1. 프로젝트 구조

```
mediassist-ai/
├── .github/
│   └── workflows/
│       ├── lint.yml
│       ├── test.yml
│       ├── build.yml
│       └── deploy.yml
├── services/
│   ├── auth/
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   ├── config.py
│   │   │   ├── dependencies.py
│   │   │   ├── routers/
│   │   │   ├── schemas/
│   │   │   ├── crud/
│   │   │   └── services/
│   │   ├── tests/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── patient/
│   ├── analysis/
│   └── report/
├── models/
│   ├── symptom_classifier/
│   │   ├── __init__.py
│   │   ├── ner.py
│   │   ├── classifier.py
│   │   ├── trainer.py
│   │   └── weights/
│   ├── image_analyzer/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── gradcam.py
│   │   └── weights/
│   └── risk_predictor/
├── rag/
│   ├── __init__.py
│   ├── ingestion/
│   ├── embedding/
│   ├── retrieval/
│   ├── reranking/
│   └── generation/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   ├── services/
│   │   ├── stores/
│   │   └── types/
│   ├── public/
│   └── package.json
├── common/
│   ├── __init__.py
│   ├── database.py
│   ├── models/
│   ├── schemas/
│   └── utils/
├── tests/
├── k8s/
├── docs/
├── scripts/
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── README.md
```

---

## 2. Backend 구현

### 2.1 FastAPI 서비스 기본 구조

#### main.py

```python
# services/analysis/app/main.py

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import symptoms, images, rag
from app.dependencies import init_ml_models, init_rag_system
from common.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작/종료 시 실행"""
    # Startup
    await init_db()
    await init_ml_models()
    await init_rag_system()
    yield
    # Shutdown
    pass


app = FastAPI(
    title="MediAssist Analysis Service",
    description="Medical RAG + ML Analysis API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(symptoms.router, prefix="/api/v1/analysis", tags=["Symptoms"])
app.include_router(images.router, prefix="/api/v1/analysis", tags=["Images"])
app.include_router(rag.router, prefix="/api/v1/rag", tags=["RAG"])


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/ready")
async def readiness_check():
    return {"status": "ready", "ml_models": True, "rag": True}
```

#### config.py

```python
# services/analysis/app/config.py

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # App
    APP_NAME: str = "MediAssist Analysis Service"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost:5432/mediassist"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # ML Models
    SYMPTOM_MODEL_PATH: str = "./models/symptom_classifier/weights"
    IMAGE_MODEL_PATH: str = "./models/image_analyzer/weights"
    
    # RAG
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    OPENAI_API_KEY: str = ""
    
    # Security
    JWT_SECRET: str = "your-secret-key"
    JWT_ALGORITHM: str = "HS256"
    
    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
```

#### schemas/analysis.py

```python
# services/analysis/app/schemas/analysis.py

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from uuid import UUID
from enum import Enum


class AnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SymptomInput(BaseModel):
    """증상 입력 스키마"""
    name: str = Field(..., min_length=1, max_length=200)
    severity: Optional[int] = Field(None, ge=1, le=10)
    duration_days: Optional[int] = Field(None, ge=0)
    description: Optional[str] = Field(None, max_length=1000)


class PatientInfo(BaseModel):
    """환자 정보 스키마"""
    age: Optional[int] = Field(None, ge=0, le=150)
    gender: Optional[str] = Field(None, pattern="^(male|female|other)$")
    medical_history: list[str] = Field(default_factory=list)


class SymptomAnalysisRequest(BaseModel):
    """증상 분석 요청"""
    patient_id: Optional[UUID] = None
    symptoms: list[SymptomInput] = Field(..., min_length=1)
    patient_info: Optional[PatientInfo] = None
    include_rag: bool = True
    include_risk_assessment: bool = True


class PredictionResult(BaseModel):
    """예측 결과"""
    disease: str
    icd_code: str
    probability: float = Field(..., ge=0, le=1)
    confidence: str = Field(..., pattern="^(high|medium|low)$")


class RiskAssessment(BaseModel):
    """위험도 평가"""
    score: float = Field(..., ge=0, le=100)
    level: str = Field(..., pattern="^(low|moderate|high|critical)$")
    factors: list[str] = Field(default_factory=list)


class SourceReference(BaseModel):
    """문헌 참조"""
    title: str
    authors: Optional[str] = None
    year: Optional[int] = None
    journal: Optional[str] = None
    relevance: float = Field(..., ge=0, le=1)
    url: Optional[str] = None


class RAGResult(BaseModel):
    """RAG 결과"""
    query: str
    summary: str
    recommendations: list[str] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    """분석 응답"""
    id: UUID
    status: AnalysisStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    processing_time_ms: Optional[int] = None
    
    predictions: Optional[list[PredictionResult]] = None
    risk_assessment: Optional[RiskAssessment] = None
    rag_result: Optional[RAGResult] = None
    
    disclaimer: str = "이 분석 결과는 참고용이며, 최종 진단은 의사가 결정해야 합니다."
```

#### routers/symptoms.py

```python
# services/analysis/app/routers/symptoms.py

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from uuid import UUID, uuid4

from app.schemas.analysis import (
    SymptomAnalysisRequest,
    AnalysisResponse,
    AnalysisStatus,
)
from app.dependencies import get_current_user, get_ml_service, get_rag_service
from app.services.ml_service import MLService
from app.services.rag_service import RAGService
from common.models import User


router = APIRouter()


@router.post("/symptoms", response_model=dict, status_code=202)
async def create_symptom_analysis(
    request: SymptomAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    ml_service: MLService = Depends(get_ml_service),
):
    """증상 분석 요청 생성"""
    analysis_id = uuid4()
    
    # 비동기 분석 태스크 등록
    background_tasks.add_task(
        process_symptom_analysis,
        analysis_id=analysis_id,
        request=request,
        user_id=current_user.id,
    )
    
    return {
        "analysis_id": str(analysis_id),
        "status": "processing",
        "estimated_time_seconds": 30,
    }


@router.get("/symptoms/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis_result(
    analysis_id: UUID,
    current_user: User = Depends(get_current_user),
):
    """분석 결과 조회"""
    # DB에서 분석 결과 조회
    analysis = await get_analysis_by_id(analysis_id)
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # 권한 확인
    if analysis.created_by != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    return analysis


async def process_symptom_analysis(
    analysis_id: UUID,
    request: SymptomAnalysisRequest,
    user_id: UUID,
):
    """증상 분석 처리 (Background Task)"""
    from app.dependencies import get_ml_service, get_rag_service
    
    ml_service = get_ml_service()
    rag_service = get_rag_service()
    
    try:
        # 1. ML 분석
        ml_result = await ml_service.analyze_symptoms(
            symptoms=request.symptoms,
            patient_info=request.patient_info,
        )
        
        # 2. RAG 검색 (옵션)
        rag_result = None
        if request.include_rag and ml_result.predictions:
            top_disease = ml_result.predictions[0].disease
            rag_result = await rag_service.query(
                question=f"{top_disease} 진단 및 치료",
                context={"symptoms": [s.name for s in request.symptoms]}
            )
        
        # 3. 결과 저장
        await save_analysis_result(
            analysis_id=analysis_id,
            user_id=user_id,
            ml_result=ml_result,
            rag_result=rag_result,
            status=AnalysisStatus.COMPLETED,
        )
        
    except Exception as e:
        await save_analysis_result(
            analysis_id=analysis_id,
            user_id=user_id,
            status=AnalysisStatus.FAILED,
            error_message=str(e),
        )
```

---

## 3. ML 모델 구현

### 3.1 증상 분류 모델

```python
# models/symptom_classifier/classifier.py

from typing import List, Dict, Optional
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import AutoTokenizer, AutoModel
import joblib


class MedicalNER:
    """의료 엔티티 인식"""
    
    def __init__(self, model_name: str = "dmis-lab/biobert-v1.1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def get_embeddings(self, text: str) -> np.ndarray:
        """텍스트 임베딩 추출"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy().flatten()


class SymptomClassifier:
    """증상 기반 질병 분류"""
    
    def __init__(self, model_path: str = None):
        self.ner = MedicalNER()
        self.label_encoder = LabelEncoder()
        self.ensemble = None
        
        if model_path:
            self.load(model_path)
    
    def _extract_features(
        self,
        symptoms: List[Dict],
        patient_info: Optional[Dict] = None,
    ) -> np.ndarray:
        """특징 벡터 추출"""
        # 증상 텍스트 결합
        symptom_text = " ".join([s["name"] for s in symptoms])
        
        # BioBERT 임베딩
        bert_embedding = self.ner.get_embeddings(symptom_text)
        
        # 환자 정보 특징
        patient_features = np.zeros(5)
        if patient_info:
            patient_features[0] = patient_info.get("age", 0) / 100
            patient_features[1] = 1 if patient_info.get("gender") == "male" else 0
            patient_features[2] = len(patient_info.get("medical_history", [])) / 10
        
        # 증상 수치 특징
        symptom_features = np.array([
            np.mean([s.get("severity", 5) for s in symptoms]) / 10,
            np.mean([s.get("duration_days", 1) for s in symptoms]) / 30,
            len(symptoms) / 10,
        ])
        
        return np.concatenate([bert_embedding, patient_features, symptom_features])
    
    def train(self, X: List[Dict], y: List[str]) -> Dict[str, float]:
        """모델 학습"""
        # 특징 추출
        X_features = np.array([
            self._extract_features(x["symptoms"], x.get("patient_info"))
            for x in X
        ])
        
        # 레이블 인코딩
        y_encoded = self.label_encoder.fit_transform(y)
        
        # 앙상블 모델
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob",
            random_state=42,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
        
        rf_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
        )
        
        self.ensemble = VotingClassifier(
            estimators=[("xgb", xgb_clf), ("rf", rf_clf)],
            voting="soft",
            weights=[0.6, 0.4],
        )
        
        self.ensemble.fit(X_features, y_encoded)
        
        # 평가
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(self.ensemble, X_features, y_encoded, cv=5)
        
        return {
            "accuracy_mean": scores.mean(),
            "accuracy_std": scores.std(),
        }
    
    def predict(
        self,
        symptoms: List[Dict],
        patient_info: Optional[Dict] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        """예측"""
        features = self._extract_features(symptoms, patient_info).reshape(1, -1)
        probabilities = self.ensemble.predict_proba(features)[0]
        
        # 상위 k개
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        
        predictions = []
        for idx in top_indices:
            disease = self.label_encoder.inverse_transform([idx])[0]
            prob = probabilities[idx]
            
            predictions.append({
                "disease": disease,
                "icd_code": self._get_icd_code(disease),
                "probability": float(prob),
                "confidence": self._get_confidence(prob),
            })
        
        return predictions
    
    def _get_confidence(self, prob: float) -> str:
        if prob >= 0.7:
            return "high"
        elif prob >= 0.4:
            return "medium"
        return "low"
    
    def _get_icd_code(self, disease: str) -> str:
        # ICD-10 매핑 (실제 구현 필요)
        icd_mapping = {
            "급성 상기도 감염": "J06.9",
            "인플루엔자": "J11",
            "폐렴": "J18.9",
        }
        return icd_mapping.get(disease, "R69")
    
    def save(self, path: str):
        """모델 저장"""
        joblib.dump({
            "ensemble": self.ensemble,
            "label_encoder": self.label_encoder,
        }, f"{path}/classifier.pkl")
    
    def load(self, path: str):
        """모델 로드"""
        data = joblib.load(f"{path}/classifier.pkl")
        self.ensemble = data["ensemble"]
        self.label_encoder = data["label_encoder"]
```

### 3.2 이미지 분석 모델

```python
# models/image_analyzer/model.py

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Dict, List
import cv2


class ChestXrayAnalyzer(nn.Module):
    """흉부 X-ray 분석 모델"""
    
    CONDITIONS = [
        "정상", "폐렴", "결핵", "폐암 의심",
        "심비대", "기흉", "폐부종"
    ]
    
    def __init__(self, num_classes: int = 7):
        super().__init__()
        
        self.backbone = models.densenet121(pretrained=True)
        num_features = self.backbone.classifier.in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.Sigmoid(),
        )
        
        self.gradients = None
        self.activations = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def activations_hook(self, grad):
        self.gradients = grad


class ImageAnalyzer:
    """이미지 분석 서비스"""
    
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChestXrayAnalyzer().to(self.device)
        
        if model_path:
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
        
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    
    def analyze(self, image_path: str) -> Dict:
        """이미지 분석"""
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        probabilities = outputs.cpu().numpy()[0]
        
        findings = []
        for i, (condition, prob) in enumerate(
            zip(ChestXrayAnalyzer.CONDITIONS, probabilities)
        ):
            if prob > 0.3:
                findings.append({
                    "condition": condition,
                    "probability": float(prob),
                    "confidence": "high" if prob > 0.7 else "medium" if prob > 0.5 else "low",
                })
        
        findings.sort(key=lambda x: x["probability"], reverse=True)
        
        return {
            "findings": findings,
            "image_quality": self._assess_quality(image_path),
        }
    
    def _assess_quality(self, image_path: str) -> Dict:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return {
            "resolution": image.shape,
            "brightness": float(np.mean(image)),
            "contrast": float(np.std(image)),
            "is_acceptable": np.std(image) > 30,
        }
```

---

## 4. RAG 시스템 구현

```python
# rag/medical_rag.py

from typing import List, Dict, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from sentence_transformers import CrossEncoder


class MedicalRAG:
    """의료 RAG 시스템"""
    
    PROMPT_TEMPLATE = """당신은 의료 전문가를 위한 진단 보조 AI입니다.

## 참고 문헌
{context}

## 질문
{question}

## 지침
1. 문헌에 근거하여 답변하세요
2. 불확실한 경우 명시하세요
3. 참고문헌을 [1], [2] 형식으로 인용하세요
4. 이 정보는 참고용임을 명시하세요

## 답변"""
    
    def __init__(
        self,
        embedding_model: str = "dmis-lab/biobert-v1.1",
        chroma_persist_dir: str = "./chroma_db",
        openai_api_key: str = None,
    ):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cuda"},
        )
        
        self.vectorstore = Chroma(
            persist_directory=chroma_persist_dir,
            embedding_function=self.embeddings,
        )
        
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,
            openai_api_key=openai_api_key,
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
    
    def add_documents(self, documents: List[Dict]) -> int:
        """문서 추가"""
        texts = []
        metadatas = []
        
        for doc in documents:
            chunks = self.text_splitter.split_text(doc["content"])
            for chunk in chunks:
                texts.append(chunk)
                metadatas.append({
                    "source": doc.get("source", ""),
                    "title": doc.get("title", ""),
                    "authors": doc.get("authors", ""),
                    "year": doc.get("year", ""),
                })
        
        self.vectorstore.add_texts(texts, metadatas=metadatas)
        return len(texts)
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """검색 + 재순위화"""
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        
        if not results:
            return []
        
        pairs = [[query, doc.page_content] for doc, _ in results]
        scores = self.reranker.predict(pairs)
        
        reranked = sorted(
            zip(results, scores),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
            }
            for (doc, _), score in reranked
        ]
    
    def query(self, question: str, context: Dict = None) -> Dict:
        """RAG 쿼리"""
        expanded = question
        if context and context.get("symptoms"):
            expanded += f" 증상: {', '.join(context['symptoms'])}"
        
        docs = self.search(expanded)
        
        if not docs:
            return {
                "answer": "관련 문헌을 찾을 수 없습니다.",
                "sources": [],
            }
        
        context_text = "\n\n".join([
            f"[{i+1}] {d['content']}"
            for i, d in enumerate(docs)
        ])
        
        prompt = PromptTemplate(
            template=self.PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )
        
        chain = prompt | self.llm
        result = chain.invoke({"context": context_text, "question": question})
        
        return {
            "answer": result.content,
            "sources": [
                {
                    "title": d["metadata"].get("title"),
                    "authors": d["metadata"].get("authors"),
                    "year": d["metadata"].get("year"),
                    "relevance": d["score"],
                }
                for d in docs
            ],
        }
```

---

## 5. Frontend 구현

### 5.1 API 클라이언트

```typescript
// frontend/src/services/api/client.ts

import axios, { AxiosInstance, AxiosError } from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.client.interceptors.request.use((config) => {
      const token = localStorage.getItem('access_token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });

    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        if (error.response?.status === 401) {
          localStorage.removeItem('access_token');
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  // Auth
  async login(email: string, password: string) {
    const response = await this.client.post('/api/v1/auth/login', {
      email,
      password,
    });
    return response.data;
  }

  // Analysis
  async createSymptomAnalysis(data: SymptomAnalysisRequest) {
    const response = await this.client.post('/api/v1/analysis/symptoms', data);
    return response.data;
  }

  async getAnalysisResult(analysisId: string) {
    const response = await this.client.get(`/api/v1/analysis/symptoms/${analysisId}`);
    return response.data;
  }

  async uploadImage(file: File, imageType: string) {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('image_type', imageType);

    const response = await this.client.post('/api/v1/analysis/image', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  }
}

export const apiClient = new ApiClient();
```

### 5.2 증상 입력 컴포넌트

```tsx
// frontend/src/pages/analysis/SymptomInput.tsx

import React, { useState } from 'react';
import { useForm, useFieldArray } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useMutation } from '@tanstack/react-query';
import { apiClient } from '@/services/api/client';

const symptomSchema = z.object({
  symptoms: z.array(z.object({
    name: z.string().min(1, '증상명을 입력하세요'),
    severity: z.number().min(1).max(10).optional(),
    duration_days: z.number().min(0).optional(),
    description: z.string().optional(),
  })).min(1, '최소 1개 이상의 증상을 입력하세요'),
  patient_info: z.object({
    age: z.number().optional(),
    gender: z.enum(['male', 'female', 'other']).optional(),
    medical_history: z.array(z.string()).optional(),
  }).optional(),
});

type SymptomFormData = z.infer<typeof symptomSchema>;

export function SymptomInput() {
  const [analysisId, setAnalysisId] = useState<string | null>(null);

  const { register, control, handleSubmit, formState: { errors } } = useForm<SymptomFormData>({
    resolver: zodResolver(symptomSchema),
    defaultValues: {
      symptoms: [{ name: '', severity: 5, duration_days: 1 }],
    },
  });

  const { fields, append, remove } = useFieldArray({
    control,
    name: 'symptoms',
  });

  const mutation = useMutation({
    mutationFn: (data: SymptomFormData) => apiClient.createSymptomAnalysis(data),
    onSuccess: (data) => {
      setAnalysisId(data.analysis_id);
    },
  });

  const onSubmit = (data: SymptomFormData) => {
    mutation.mutate(data);
  };

  return (
    <div className="max-w-2xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">증상 분석</h1>

      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
        {/* 증상 입력 */}
        <div>
          <label className="block text-sm font-medium mb-2">증상</label>
          {fields.map((field, index) => (
            <div key={field.id} className="flex gap-4 mb-4">
              <input
                {...register(`symptoms.${index}.name`)}
                placeholder="증상명"
                className="flex-1 border rounded px-3 py-2"
              />
              <input
                type="number"
                {...register(`symptoms.${index}.severity`, { valueAsNumber: true })}
                placeholder="심각도 (1-10)"
                className="w-32 border rounded px-3 py-2"
              />
              <button
                type="button"
                onClick={() => remove(index)}
                className="text-red-500"
              >
                삭제
              </button>
            </div>
          ))}
          <button
            type="button"
            onClick={() => append({ name: '', severity: 5 })}
            className="text-blue-500"
          >
            + 증상 추가
          </button>
        </div>

        {/* 제출 버튼 */}
        <button
          type="submit"
          disabled={mutation.isPending}
          className="w-full bg-blue-600 text-white py-3 rounded hover:bg-blue-700"
        >
          {mutation.isPending ? '분석 중...' : '분석 시작'}
        </button>
      </form>

      {analysisId && (
        <div className="mt-6 p-4 bg-green-100 rounded">
          분석이 시작되었습니다. ID: {analysisId}
        </div>
      )}
    </div>
  );
}
```

---

## 6. Docker 및 배포

### 6.1 Dockerfile

```dockerfile
# services/analysis/Dockerfile

FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드
COPY . .

# 포트
EXPOSE 8000

# 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 6.2 docker-compose.yml

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: mediassist
      POSTGRES_PASSWORD: password
      POSTGRES_DB: mediassist
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8005:8000"
    volumes:
      - chroma_data:/chroma/chroma

  analysis-service:
    build: ./services/analysis
    environment:
      - DATABASE_URL=postgresql+asyncpg://mediassist:password@postgres:5432/mediassist
      - REDIS_URL=redis://redis:6379/0
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    ports:
      - "8003:8000"
    depends_on:
      - postgres
      - redis
      - chromadb

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - VITE_API_URL=http://localhost:8003

volumes:
  postgres_data:
  chroma_data:
```

---

## 7. 테스트

### 7.1 단위 테스트

```python
# tests/test_symptom_classifier.py

import pytest
from models.symptom_classifier.classifier import SymptomClassifier


@pytest.fixture
def classifier():
    return SymptomClassifier()


@pytest.fixture
def sample_symptoms():
    return [
        {"name": "두통", "severity": 7, "duration_days": 3},
        {"name": "발열", "severity": 8, "duration_days": 2},
    ]


def test_predict_returns_list(classifier, sample_symptoms):
    predictions = classifier.predict(sample_symptoms)
    assert isinstance(predictions, list)
    assert len(predictions) <= 5


def test_prediction_format(classifier, sample_symptoms):
    predictions = classifier.predict(sample_symptoms)
    for pred in predictions:
        assert "disease" in pred
        assert "probability" in pred
        assert 0 <= pred["probability"] <= 1
```

---

**END OF IMPLEMENTATION**
