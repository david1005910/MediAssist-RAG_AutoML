"""Initialize Neo4j with comprehensive medical knowledge graph based on SPECIFICATION.md."""
from neo4j import GraphDatabase

# Connection settings
URI = "bolt://localhost:7688"
AUTH = ("neo4j", "password123")


def init_graph(driver):
    """Initialize the medical knowledge graph."""

    with driver.session() as session:
        # Clear existing data
        print("Clearing existing data...")
        session.run("MATCH (n) DETACH DELETE n")

        # ============================================================
        # 1. Create Disease nodes with ICD codes (from SPECIFICATION.md)
        # ============================================================
        print("Creating Disease nodes...")
        diseases = [
            # 호흡기 질환
            {"name": "급성 상기도 감염", "name_en": "Acute Upper Respiratory Infection", "icd_code": "J06.9", "category": "호흡기", "risk_level": "low"},
            {"name": "인플루엔자", "name_en": "Influenza", "icd_code": "J11", "category": "호흡기", "risk_level": "medium"},
            {"name": "폐렴", "name_en": "Pneumonia", "icd_code": "J18.9", "category": "호흡기", "risk_level": "high"},
            {"name": "폐결핵", "name_en": "Pulmonary Tuberculosis", "icd_code": "A15", "category": "호흡기", "risk_level": "high"},
            {"name": "폐암", "name_en": "Lung Cancer", "icd_code": "C34", "category": "호흡기", "risk_level": "critical"},
            {"name": "천식", "name_en": "Asthma", "icd_code": "J45", "category": "호흡기", "risk_level": "medium"},
            {"name": "기흉", "name_en": "Pneumothorax", "icd_code": "J93", "category": "호흡기", "risk_level": "high"},
            {"name": "폐부종", "name_en": "Pulmonary Edema", "icd_code": "J81", "category": "호흡기", "risk_level": "critical"},
            {"name": "COPD", "name_en": "Chronic Obstructive Pulmonary Disease", "icd_code": "J44", "category": "호흡기", "risk_level": "high"},
            # 심혈관 질환
            {"name": "심부전", "name_en": "Heart Failure", "icd_code": "I50", "category": "심혈관", "risk_level": "high"},
            {"name": "고혈압", "name_en": "Hypertension", "icd_code": "I10", "category": "심혈관", "risk_level": "medium"},
            {"name": "심비대", "name_en": "Cardiomegaly", "icd_code": "I51.7", "category": "심혈관", "risk_level": "medium"},
            {"name": "협심증", "name_en": "Angina Pectoris", "icd_code": "I20", "category": "심혈관", "risk_level": "high"},
            {"name": "심근경색", "name_en": "Myocardial Infarction", "icd_code": "I21", "category": "심혈관", "risk_level": "critical"},
            # 대사 질환
            {"name": "당뇨병", "name_en": "Diabetes Mellitus", "icd_code": "E11", "category": "대사", "risk_level": "medium"},
            {"name": "고지혈증", "name_en": "Hyperlipidemia", "icd_code": "E78", "category": "대사", "risk_level": "low"},
        ]

        for d in diseases:
            session.run("""
                CREATE (d:Disease {
                    name: $name,
                    name_en: $name_en,
                    icd_code: $icd_code,
                    category: $category,
                    risk_level: $risk_level
                })
            """, d)

        # ============================================================
        # 2. Create Symptom nodes (from SPECIFICATION.md analysis)
        # ============================================================
        print("Creating Symptom nodes...")
        symptoms = [
            {"name": "발열", "name_en": "Fever", "category": "전신"},
            {"name": "기침", "name_en": "Cough", "category": "호흡기"},
            {"name": "가래", "name_en": "Sputum", "category": "호흡기"},
            {"name": "호흡곤란", "name_en": "Dyspnea", "category": "호흡기"},
            {"name": "흉통", "name_en": "Chest Pain", "category": "흉부"},
            {"name": "두통", "name_en": "Headache", "category": "신경"},
            {"name": "근육통", "name_en": "Myalgia", "category": "근골격"},
            {"name": "피로감", "name_en": "Fatigue", "category": "전신"},
            {"name": "체중감소", "name_en": "Weight Loss", "category": "전신"},
            {"name": "야간발한", "name_en": "Night Sweats", "category": "전신"},
            {"name": "객혈", "name_en": "Hemoptysis", "category": "호흡기"},
            {"name": "인후통", "name_en": "Sore Throat", "category": "호흡기"},
            {"name": "콧물", "name_en": "Rhinorrhea", "category": "호흡기"},
            {"name": "코막힘", "name_en": "Nasal Congestion", "category": "호흡기"},
            {"name": "부종", "name_en": "Edema", "category": "순환"},
            {"name": "현기증", "name_en": "Dizziness", "category": "신경"},
            {"name": "오심", "name_en": "Nausea", "category": "소화기"},
            {"name": "식욕부진", "name_en": "Anorexia", "category": "소화기"},
            {"name": "청색증", "name_en": "Cyanosis", "category": "순환"},
            {"name": "빈맥", "name_en": "Tachycardia", "category": "심혈관"},
        ]

        for s in symptoms:
            session.run("""
                CREATE (s:Symptom {
                    name: $name,
                    name_en: $name_en,
                    category: $category
                })
            """, s)

        # ============================================================
        # 3. Create Treatment nodes
        # ============================================================
        print("Creating Treatment nodes...")
        treatments = [
            {"name": "항생제 치료", "name_en": "Antibiotic Therapy", "type": "약물치료"},
            {"name": "항바이러스제", "name_en": "Antiviral Therapy", "type": "약물치료"},
            {"name": "기관지확장제", "name_en": "Bronchodilator", "type": "약물치료"},
            {"name": "스테로이드 치료", "name_en": "Corticosteroid Therapy", "type": "약물치료"},
            {"name": "항결핵제", "name_en": "Anti-TB Therapy", "type": "약물치료"},
            {"name": "산소 치료", "name_en": "Oxygen Therapy", "type": "지지요법"},
            {"name": "이뇨제", "name_en": "Diuretics", "type": "약물치료"},
            {"name": "ACE억제제", "name_en": "ACE Inhibitor", "type": "약물치료"},
            {"name": "베타차단제", "name_en": "Beta Blocker", "type": "약물치료"},
            {"name": "칼슘채널차단제", "name_en": "Calcium Channel Blocker", "type": "약물치료"},
            {"name": "화학요법", "name_en": "Chemotherapy", "type": "종양치료"},
            {"name": "방사선치료", "name_en": "Radiation Therapy", "type": "종양치료"},
            {"name": "수액치료", "name_en": "IV Fluid Therapy", "type": "지지요법"},
            {"name": "해열제", "name_en": "Antipyretic", "type": "대증치료"},
            {"name": "인슐린", "name_en": "Insulin", "type": "약물치료"},
            {"name": "흉관삽입술", "name_en": "Chest Tube Insertion", "type": "시술"},
            {"name": "대증 치료", "name_en": "Symptomatic Treatment", "type": "대증치료"},
        ]

        for t in treatments:
            session.run("""
                CREATE (t:Treatment {
                    name: $name,
                    name_en: $name_en,
                    type: $type
                })
            """, t)

        # ============================================================
        # 4. Create Drug nodes
        # ============================================================
        print("Creating Drug nodes...")
        drugs = [
            {"name": "아목시실린", "name_en": "Amoxicillin", "class": "페니실린계 항생제"},
            {"name": "아지스로마이신", "name_en": "Azithromycin", "class": "마크로라이드 항생제"},
            {"name": "레보플록사신", "name_en": "Levofloxacin", "class": "퀴놀론 항생제"},
            {"name": "오셀타미비르", "name_en": "Oseltamivir", "class": "항바이러스제"},
            {"name": "살부타몰", "name_en": "Salbutamol", "class": "기관지확장제"},
            {"name": "프레드니손", "name_en": "Prednisone", "class": "코르티코스테로이드"},
            {"name": "이소니아지드", "name_en": "Isoniazid", "class": "항결핵제"},
            {"name": "리팜핀", "name_en": "Rifampin", "class": "항결핵제"},
            {"name": "푸로세미드", "name_en": "Furosemide", "class": "루프이뇨제"},
            {"name": "리시노프릴", "name_en": "Lisinopril", "class": "ACE억제제"},
            {"name": "암로디핀", "name_en": "Amlodipine", "class": "칼슘채널차단제"},
            {"name": "메토프롤롤", "name_en": "Metoprolol", "class": "베타차단제"},
            {"name": "아세트아미노펜", "name_en": "Acetaminophen", "class": "해열진통제"},
            {"name": "메트포르민", "name_en": "Metformin", "class": "경구혈당강하제"},
            {"name": "아토르바스타틴", "name_en": "Atorvastatin", "class": "스타틴"},
        ]

        for dr in drugs:
            session.run("""
                CREATE (dr:Drug {
                    name: $name,
                    name_en: $name_en,
                    class: $class
                })
            """, dr)

        # ============================================================
        # 5. Create DiagnosticTest nodes (from SPECIFICATION.md)
        # ============================================================
        print("Creating DiagnosticTest nodes...")
        tests = [
            {"name": "흉부 X-ray", "name_en": "Chest X-ray", "type": "영상검사", "image_type": "chest_xray"},
            {"name": "흉부 CT", "name_en": "Chest CT", "type": "영상검사", "image_type": "ct_chest"},
            {"name": "두부 CT", "name_en": "Head CT", "type": "영상검사", "image_type": "ct_head"},
            {"name": "복부 CT", "name_en": "Abdominal CT", "type": "영상검사", "image_type": "ct_abdomen"},
            {"name": "심전도", "name_en": "ECG", "type": "기능검사", "image_type": None},
            {"name": "심초음파", "name_en": "Echocardiogram", "type": "영상검사", "image_type": None},
            {"name": "폐기능검사", "name_en": "Pulmonary Function Test", "type": "기능검사", "image_type": None},
            {"name": "객담검사", "name_en": "Sputum Test", "type": "검체검사", "image_type": None},
            {"name": "혈액검사", "name_en": "Blood Test", "type": "검체검사", "image_type": None},
            {"name": "혈액배양검사", "name_en": "Blood Culture", "type": "검체검사", "image_type": None},
            {"name": "소변검사", "name_en": "Urinalysis", "type": "검체검사", "image_type": None},
            {"name": "인플루엔자 신속항원검사", "name_en": "Rapid Influenza Test", "type": "검체검사", "image_type": None},
            {"name": "COVID-19 PCR", "name_en": "COVID-19 PCR", "type": "검체검사", "image_type": None},
            {"name": "동맥혈가스분석", "name_en": "ABGA", "type": "검체검사", "image_type": None},
        ]

        for te in tests:
            if te["image_type"]:
                session.run("""
                    CREATE (te:DiagnosticTest {
                        name: $name,
                        name_en: $name_en,
                        type: $type,
                        image_type: $image_type
                    })
                """, te)
            else:
                session.run("""
                    CREATE (te:DiagnosticTest {
                        name: $name,
                        name_en: $name_en,
                        type: $type
                    })
                """, {"name": te["name"], "name_en": te["name_en"], "type": te["type"]})

        # ============================================================
        # 6. Create ImageFinding nodes (X-ray analysis from SPECIFICATION.md)
        # ============================================================
        print("Creating ImageFinding nodes...")
        findings = [
            {"name": "정상", "name_en": "Normal", "description": "이상 소견 없음"},
            {"name": "폐렴 소견", "name_en": "Pneumonia", "description": "폐포 침윤, 기관지 공기음영"},
            {"name": "결핵 소견", "name_en": "Tuberculosis", "description": "상엽 침윤, 공동 형성"},
            {"name": "폐암 의심", "name_en": "Lung Cancer Suspected", "description": "종괴 음영, 결절"},
            {"name": "심비대", "name_en": "Cardiomegaly", "description": "심장 크기 증가"},
            {"name": "기흉", "name_en": "Pneumothorax", "description": "흉막강 내 공기"},
            {"name": "폐부종", "name_en": "Pulmonary Edema", "description": "폐문 주위 혼탁"},
            {"name": "늑막삼출", "name_en": "Pleural Effusion", "description": "늑막강 내 액체"},
        ]

        for f in findings:
            session.run("""
                CREATE (f:ImageFinding {
                    name: $name,
                    name_en: $name_en,
                    description: $description
                })
            """, f)

        # ============================================================
        # 7. Create RiskLevel nodes
        # ============================================================
        print("Creating RiskLevel nodes...")
        risk_levels = [
            {"level": "low", "name": "저위험", "score_range": "0-25", "action": "외래 치료"},
            {"level": "medium", "name": "중위험", "score_range": "26-50", "action": "단기 입원 고려"},
            {"level": "high", "name": "고위험", "score_range": "51-75", "action": "입원 치료"},
            {"level": "critical", "name": "최고위험", "score_range": "76-100", "action": "중환자실 고려"},
        ]

        for r in risk_levels:
            session.run("""
                CREATE (r:RiskLevel {
                    level: $level,
                    name: $name,
                    score_range: $score_range,
                    action: $action
                })
            """, r)

        # ============================================================
        # 8. Create Relationships
        # ============================================================
        print("Creating relationships...")

        # Disease -> HAS_SYMPTOM -> Symptom
        disease_symptoms = [
            # 급성 상기도 감염
            ("급성 상기도 감염", "콧물", 0.9),
            ("급성 상기도 감염", "코막힘", 0.85),
            ("급성 상기도 감염", "인후통", 0.8),
            ("급성 상기도 감염", "기침", 0.75),
            ("급성 상기도 감염", "발열", 0.5),
            # 인플루엔자
            ("인플루엔자", "발열", 0.95),
            ("인플루엔자", "두통", 0.85),
            ("인플루엔자", "근육통", 0.9),
            ("인플루엔자", "피로감", 0.85),
            ("인플루엔자", "기침", 0.7),
            # 폐렴
            ("폐렴", "발열", 0.9),
            ("폐렴", "기침", 0.95),
            ("폐렴", "가래", 0.85),
            ("폐렴", "호흡곤란", 0.75),
            ("폐렴", "흉통", 0.5),
            # 폐결핵
            ("폐결핵", "기침", 0.95),
            ("폐결핵", "야간발한", 0.7),
            ("폐결핵", "체중감소", 0.65),
            ("폐결핵", "객혈", 0.3),
            ("폐결핵", "발열", 0.6),
            # 폐암
            ("폐암", "기침", 0.8),
            ("폐암", "객혈", 0.4),
            ("폐암", "체중감소", 0.7),
            ("폐암", "흉통", 0.5),
            ("폐암", "호흡곤란", 0.6),
            # 천식
            ("천식", "호흡곤란", 0.95),
            ("천식", "기침", 0.8),
            # 기흉
            ("기흉", "흉통", 0.95),
            ("기흉", "호흡곤란", 0.9),
            # 폐부종
            ("폐부종", "호흡곤란", 0.95),
            ("폐부종", "기침", 0.7),
            ("폐부종", "청색증", 0.5),
            # 심부전
            ("심부전", "호흡곤란", 0.9),
            ("심부전", "부종", 0.85),
            ("심부전", "피로감", 0.8),
            # 고혈압
            ("고혈압", "두통", 0.4),
            ("고혈압", "현기증", 0.3),
            # 당뇨병
            ("당뇨병", "피로감", 0.6),
            ("당뇨병", "체중감소", 0.5),
        ]

        for disease, symptom, prob in disease_symptoms:
            session.run("""
                MATCH (d:Disease {name: $disease}), (s:Symptom {name: $symptom})
                CREATE (d)-[:HAS_SYMPTOM {probability: $prob}]->(s)
            """, {"disease": disease, "symptom": symptom, "prob": prob})

        # Disease -> TREATED_BY -> Treatment
        disease_treatments = [
            ("급성 상기도 감염", "대증 치료"),
            ("급성 상기도 감염", "해열제"),
            ("인플루엔자", "항바이러스제"),
            ("인플루엔자", "해열제"),
            ("폐렴", "항생제 치료"),
            ("폐렴", "산소 치료"),
            ("폐렴", "수액치료"),
            ("폐결핵", "항결핵제"),
            ("폐암", "화학요법"),
            ("폐암", "방사선치료"),
            ("천식", "기관지확장제"),
            ("천식", "스테로이드 치료"),
            ("기흉", "흉관삽입술"),
            ("기흉", "산소 치료"),
            ("폐부종", "이뇨제"),
            ("폐부종", "산소 치료"),
            ("심부전", "이뇨제"),
            ("심부전", "ACE억제제"),
            ("심부전", "베타차단제"),
            ("고혈압", "ACE억제제"),
            ("고혈압", "칼슘채널차단제"),
            ("고혈압", "베타차단제"),
            ("당뇨병", "인슐린"),
        ]

        for disease, treatment in disease_treatments:
            session.run("""
                MATCH (d:Disease {name: $disease}), (t:Treatment {name: $treatment})
                CREATE (d)-[:TREATED_BY]->(t)
            """, {"disease": disease, "treatment": treatment})

        # Treatment -> USES_DRUG -> Drug
        treatment_drugs = [
            ("항생제 치료", "아목시실린"),
            ("항생제 치료", "아지스로마이신"),
            ("항생제 치료", "레보플록사신"),
            ("항바이러스제", "오셀타미비르"),
            ("기관지확장제", "살부타몰"),
            ("스테로이드 치료", "프레드니손"),
            ("항결핵제", "이소니아지드"),
            ("항결핵제", "리팜핀"),
            ("이뇨제", "푸로세미드"),
            ("ACE억제제", "리시노프릴"),
            ("칼슘채널차단제", "암로디핀"),
            ("베타차단제", "메토프롤롤"),
            ("해열제", "아세트아미노펜"),
        ]

        for treatment, drug in treatment_drugs:
            session.run("""
                MATCH (t:Treatment {name: $treatment}), (dr:Drug {name: $drug})
                CREATE (t)-[:USES_DRUG]->(dr)
            """, {"treatment": treatment, "drug": drug})

        # Disease -> DIAGNOSED_BY -> DiagnosticTest
        disease_tests = [
            ("급성 상기도 감염", "혈액검사"),
            ("인플루엔자", "인플루엔자 신속항원검사"),
            ("인플루엔자", "혈액검사"),
            ("폐렴", "흉부 X-ray"),
            ("폐렴", "혈액검사"),
            ("폐렴", "객담검사"),
            ("폐렴", "혈액배양검사"),
            ("폐결핵", "흉부 X-ray"),
            ("폐결핵", "객담검사"),
            ("폐결핵", "흉부 CT"),
            ("폐암", "흉부 X-ray"),
            ("폐암", "흉부 CT"),
            ("천식", "폐기능검사"),
            ("기흉", "흉부 X-ray"),
            ("폐부종", "흉부 X-ray"),
            ("폐부종", "심초음파"),
            ("심부전", "심초음파"),
            ("심부전", "심전도"),
            ("심부전", "흉부 X-ray"),
            ("고혈압", "혈액검사"),
            ("당뇨병", "혈액검사"),
        ]

        for disease, test in disease_tests:
            session.run("""
                MATCH (d:Disease {name: $disease}), (te:DiagnosticTest {name: $test})
                CREATE (d)-[:DIAGNOSED_BY]->(te)
            """, {"disease": disease, "test": test})

        # Disease -> HAS_RISK_LEVEL -> RiskLevel
        for disease_data in diseases:
            session.run("""
                MATCH (d:Disease {name: $name}), (r:RiskLevel {level: $level})
                CREATE (d)-[:HAS_RISK_LEVEL]->(r)
            """, {"name": disease_data["name"], "level": disease_data["risk_level"]})

        # ImageFinding -> INDICATES -> Disease
        finding_diseases = [
            ("정상", None),
            ("폐렴 소견", "폐렴"),
            ("결핵 소견", "폐결핵"),
            ("폐암 의심", "폐암"),
            ("심비대", "심비대"),
            ("기흉", "기흉"),
            ("폐부종", "폐부종"),
        ]

        for finding, disease in finding_diseases:
            if disease:
                session.run("""
                    MATCH (f:ImageFinding {name: $finding}), (d:Disease {name: $disease})
                    CREATE (f)-[:INDICATES]->(d)
                """, {"finding": finding, "disease": disease})

        # DiagnosticTest -> DETECTS -> ImageFinding (for imaging tests)
        session.run("""
            MATCH (te:DiagnosticTest {name: '흉부 X-ray'}), (f:ImageFinding)
            CREATE (te)-[:DETECTS]->(f)
        """)

        print("Knowledge graph initialization complete!")


def verify_graph(driver):
    """Verify the created graph."""
    with driver.session() as session:
        # Node counts
        result = session.run("""
            MATCH (n)
            RETURN labels(n)[0] as label, count(*) as count
            ORDER BY count DESC
        """)
        print("\n=== Node Counts ===")
        for record in result:
            print(f"  {record['label']}: {record['count']}")

        # Relationship counts
        result = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) as type, count(*) as count
            ORDER BY count DESC
        """)
        print("\n=== Relationship Counts ===")
        for record in result:
            print(f"  {record['type']}: {record['count']}")

        # Sample queries
        print("\n=== Sample Query: 폐렴 관련 정보 ===")
        result = session.run("""
            MATCH (d:Disease {name: '폐렴'})-[r]->(n)
            RETURN type(r) as relationship, labels(n)[0] as target_type, n.name as target_name
            LIMIT 10
        """)
        for record in result:
            print(f"  폐렴 --[{record['relationship']}]--> {record['target_type']}: {record['target_name']}")


def main():
    print("Connecting to Neo4j...")
    driver = GraphDatabase.driver(URI, auth=AUTH)

    try:
        driver.verify_connectivity()
        print("Connected successfully!")

        init_graph(driver)
        verify_graph(driver)

        print("\n✅ Medical Knowledge Graph initialized successfully!")
        print("   Based on SPECIFICATION.md")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
