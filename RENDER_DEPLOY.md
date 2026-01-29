# Render 배포 가이드

## 사전 준비 (외부 서비스)

Render에서 직접 호스팅할 수 없는 서비스들을 먼저 설정해야 합니다.

### 1. MongoDB Atlas (필수)

1. [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) 계정 생성
2. 무료 클러스터 생성 (M0 Free Tier)
3. Database Access에서 사용자 생성
4. Network Access에서 `0.0.0.0/0` 허용 (또는 Render IP만 허용)
5. Connect → Drivers에서 연결 문자열 복사:
   ```
   mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```

### 2. OpenAI API (필수 - RAG 기능용)

1. [OpenAI Platform](https://platform.openai.com/)에서 API Key 발급
2. 또는 [Google AI Studio](https://aistudio.google.com/)에서 Gemini API Key 발급

### 3. Neo4j Aura (선택 - Knowledge Graph용)

1. [Neo4j Aura](https://neo4j.com/cloud/aura/) 무료 계정 생성
2. Free 인스턴스 생성
3. 연결 정보 저장:
   - URI: `neo4j+s://xxxxx.databases.neo4j.io`
   - Username: `neo4j`
   - Password: (생성 시 제공)

### 4. 파일 스토리지 (선택 - Report Service용)

S3 호환 스토리지 중 하나 선택:
- **Cloudflare R2** (10GB 무료)
- **AWS S3**
- **MinIO Cloud**

---

## Render 배포 방법

### 방법 1: Blueprint (render.yaml) 사용 (권장)

1. GitHub 저장소를 Render에 연결
2. Render Dashboard → **Blueprints** → **New Blueprint Instance**
3. 저장소 선택 후 `render.yaml` 자동 감지
4. **Apply** 클릭

### 방법 2: 수동 서비스 생성

각 서비스를 개별적으로 생성합니다.

---

## 환경변수 설정 (필수)

Blueprint 배포 후, Render Dashboard에서 다음 환경변수를 **수동으로** 설정해야 합니다:

### Analysis Service (mediassist-analysis)

| 변수 | 필수 | 설명 |
|------|------|------|
| `MONGODB_URI` | ✅ | MongoDB Atlas 연결 문자열 |
| `OPENAI_API_KEY` | ✅ | OpenAI API Key |
| `GOOGLE_API_KEY` | 선택 | Google Gemini API Key |
| `NEO4J_URI` | 선택 | Neo4j Aura URI |
| `NEO4J_PASSWORD` | 선택 | Neo4j 비밀번호 |
| `CHROMA_HOST` | 선택 | ChromaDB 호스트 |
| `MINIO_ENDPOINT` | 선택 | S3 호환 스토리지 엔드포인트 |
| `MINIO_ACCESS_KEY` | 선택 | 스토리지 Access Key |
| `MINIO_SECRET_KEY` | 선택 | 스토리지 Secret Key |

### Auth Service (mediassist-auth)

| 변수 | 필수 | 설명 |
|------|------|------|
| `MONGODB_URI` | ✅ | MongoDB Atlas 연결 문자열 (Analysis와 동일) |

### Report Service (mediassist-report)

| 변수 | 필수 | 설명 |
|------|------|------|
| `MINIO_ENDPOINT` | 선택 | S3 호환 스토리지 |
| `MINIO_ACCESS_KEY` | 선택 | 스토리지 Access Key |
| `MINIO_SECRET_KEY` | 선택 | 스토리지 Secret Key |

---

## Frontend 빌드 설정

Render에서 Docker 빌드 시 `VITE_API_URL`을 주입하려면:

1. Frontend 서비스 → **Environment** → **Build Arguments** 섹션
2. 추가:
   - Key: `VITE_API_URL`
   - Value: `https://mediassist-analysis.onrender.com`

또는 `.env.production` 파일을 프론트엔드에 추가:
```
VITE_API_URL=https://mediassist-analysis.onrender.com
```

---

## 예상 비용 (월)

| 서비스 | Plan | 비용 |
|--------|------|------|
| PostgreSQL | Starter | $7 |
| Redis | Starter | $10 |
| Analysis Service | Standard | $25 |
| Auth Service | Free | $0 |
| Patient Service | Free | $0 |
| Report Service | Free | $0 |
| Frontend | Free | $0 |
| **총계** | | **~$42/월** |

### 비용 절감 옵션

- Redis 제거: Celery 비동기 작업 비활성화 시 불필요
- Analysis Service를 Starter로 변경: 메모리 부족 시 ML 모델 로딩 실패 가능

---

## 배포 후 확인사항

### 1. Health Check 확인

```bash
# Analysis Service
curl https://mediassist-analysis.onrender.com/health

# Frontend
curl https://mediassist-frontend.onrender.com/health

# Auth Service
curl https://mediassist-auth.onrender.com/health
```

### 2. 로그인 테스트

데모 계정으로 테스트:
- Email: `demo@mediassist.ai`
- Password: `demo1234`

### 3. CORS 확인

브라우저 개발자 도구에서 CORS 오류 확인.
문제 발생 시 `CORS_ORIGINS` 환경변수 확인.

---

## 트러블슈팅

### 1. 빌드 실패

**증상**: Docker 빌드 중 메모리 부족
**해결**:
- Analysis Service를 Standard 이상 플랜으로 설정
- 또는 빌드 최적화 (multi-stage build 개선)

### 2. Health Check 실패

**증상**: 서비스가 계속 재시작됨
**해결**:
- 로그 확인: Render Dashboard → Service → Logs
- MongoDB/PostgreSQL 연결 확인
- 환경변수 설정 확인

### 3. CORS 오류

**증상**: 브라우저에서 API 호출 실패
**해결**:
- `CORS_ORIGINS`에 프론트엔드 URL 추가
- 쉼표로 구분된 다중 origin 지원:
  ```
  CORS_ORIGINS=https://mediassist-frontend.onrender.com,https://custom-domain.com
  ```

### 4. MongoDB 연결 실패

**증상**: Auth 서비스에서 인증 실패
**해결**:
- MongoDB Atlas Network Access에서 `0.0.0.0/0` 허용
- `dnspython` 패키지 설치 확인 (mongodb+srv 사용 시 필요)

### 5. ML 모델 로딩 실패

**증상**: 증상 분석/이미지 분석 실패
**해결**:
- Analysis Service 메모리 확인 (최소 2GB 권장)
- 모델 파일이 Docker 이미지에 포함되어 있는지 확인

---

## 커스텀 도메인 설정 (선택)

1. Render Dashboard → Service → **Settings** → **Custom Domain**
2. 도메인 추가 후 DNS 설정:
   - CNAME: `your-service.onrender.com`
3. 자동 SSL 인증서 발급 대기

커스텀 도메인 사용 시 `CORS_ORIGINS`와 `VITE_API_URL` 업데이트 필요.
