"""Neo4j Graph Database router for medical knowledge graph."""
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..config import settings

router = APIRouter(prefix="/api/v1/graph", tags=["Knowledge Graph"])

# Neo4j driver
_driver = None
_neo4j_available = True

try:
    from neo4j import GraphDatabase
except ImportError:
    _neo4j_available = False
    GraphDatabase = None


def get_driver():
    """Get or create Neo4j driver."""
    global _driver
    if not _neo4j_available:
        return None
    try:
        if _driver is None:
            _driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
        _driver.verify_connectivity()
        print(f"[Neo4j] Connected to {settings.NEO4J_URI}")
        return _driver
    except Exception as e:
        print(f"[Neo4j] Connection failed: {e}")
        _driver = None
        return None


# Mock data for demo when Neo4j is not available
MOCK_NODES = [
    {"id": "d1", "label": "폐렴", "type": "Disease", "properties": {"name": "폐렴", "name_en": "Pneumonia", "icd_code": "J18.9", "description": "폐에 염증이 생기는 감염성 질환"}},
    {"id": "d2", "label": "결핵", "type": "Disease", "properties": {"name": "결핵", "name_en": "Tuberculosis", "icd_code": "A15.0", "description": "결핵균에 의한 만성 감염성 질환"}},
    {"id": "d3", "label": "폐암", "type": "Disease", "properties": {"name": "폐암", "name_en": "Lung Cancer", "icd_code": "C34.9", "description": "폐에서 발생하는 악성 종양"}},
    {"id": "d4", "label": "천식", "type": "Disease", "properties": {"name": "천식", "name_en": "Asthma", "icd_code": "J45.9", "description": "기도의 만성 염증성 질환"}},
    {"id": "s1", "label": "기침", "type": "Symptom", "properties": {"name": "기침", "name_en": "Cough", "severity": "mild-severe"}},
    {"id": "s2", "label": "발열", "type": "Symptom", "properties": {"name": "발열", "name_en": "Fever", "severity": "mild-high"}},
    {"id": "s3", "label": "호흡곤란", "type": "Symptom", "properties": {"name": "호흡곤란", "name_en": "Dyspnea", "severity": "moderate-severe"}},
    {"id": "s4", "label": "흉통", "type": "Symptom", "properties": {"name": "흉통", "name_en": "Chest Pain", "severity": "mild-severe"}},
    {"id": "s5", "label": "체중감소", "type": "Symptom", "properties": {"name": "체중감소", "name_en": "Weight Loss", "severity": "gradual"}},
    {"id": "s6", "label": "야간발한", "type": "Symptom", "properties": {"name": "야간발한", "name_en": "Night Sweats", "severity": "moderate"}},
    {"id": "t1", "label": "항생제 치료", "type": "Treatment", "properties": {"name": "항생제 치료", "name_en": "Antibiotic Therapy", "type": "medication"}},
    {"id": "t2", "label": "화학요법", "type": "Treatment", "properties": {"name": "화학요법", "name_en": "Chemotherapy", "type": "medication"}},
    {"id": "t3", "label": "기관지확장제", "type": "Treatment", "properties": {"name": "기관지확장제", "name_en": "Bronchodilator", "type": "medication"}},
    {"id": "t4", "label": "항결핵제", "type": "Treatment", "properties": {"name": "항결핵제", "name_en": "Anti-TB Drugs", "type": "medication"}},
    {"id": "dr1", "label": "아목시실린", "type": "Drug", "properties": {"name": "아목시실린", "name_en": "Amoxicillin", "class": "Penicillin"}},
    {"id": "dr2", "label": "이소니아지드", "type": "Drug", "properties": {"name": "이소니아지드", "name_en": "Isoniazid", "class": "Anti-TB"}},
    {"id": "dr3", "label": "살부타몰", "type": "Drug", "properties": {"name": "살부타몰", "name_en": "Salbutamol", "class": "Beta-agonist"}},
    {"id": "dt1", "label": "흉부 X-ray", "type": "DiagnosticTest", "properties": {"name": "흉부 X-ray", "name_en": "Chest X-ray", "type": "imaging"}},
    {"id": "dt2", "label": "CT 스캔", "type": "DiagnosticTest", "properties": {"name": "CT 스캔", "name_en": "CT Scan", "type": "imaging"}},
    {"id": "dt3", "label": "객담검사", "type": "DiagnosticTest", "properties": {"name": "객담검사", "name_en": "Sputum Test", "type": "laboratory"}},
]

MOCK_EDGES = [
    # Disease - Symptom relationships
    {"source": "d1", "target": "s1", "type": "HAS_SYMPTOM", "properties": {"frequency": "common"}},
    {"source": "d1", "target": "s2", "type": "HAS_SYMPTOM", "properties": {"frequency": "common"}},
    {"source": "d1", "target": "s3", "type": "HAS_SYMPTOM", "properties": {"frequency": "common"}},
    {"source": "d2", "target": "s1", "type": "HAS_SYMPTOM", "properties": {"frequency": "common"}},
    {"source": "d2", "target": "s2", "type": "HAS_SYMPTOM", "properties": {"frequency": "common"}},
    {"source": "d2", "target": "s5", "type": "HAS_SYMPTOM", "properties": {"frequency": "common"}},
    {"source": "d2", "target": "s6", "type": "HAS_SYMPTOM", "properties": {"frequency": "characteristic"}},
    {"source": "d3", "target": "s1", "type": "HAS_SYMPTOM", "properties": {"frequency": "common"}},
    {"source": "d3", "target": "s4", "type": "HAS_SYMPTOM", "properties": {"frequency": "moderate"}},
    {"source": "d3", "target": "s5", "type": "HAS_SYMPTOM", "properties": {"frequency": "common"}},
    {"source": "d4", "target": "s1", "type": "HAS_SYMPTOM", "properties": {"frequency": "common"}},
    {"source": "d4", "target": "s3", "type": "HAS_SYMPTOM", "properties": {"frequency": "characteristic"}},
    # Disease - Treatment relationships
    {"source": "d1", "target": "t1", "type": "TREATED_BY", "properties": {"efficacy": "high"}},
    {"source": "d2", "target": "t4", "type": "TREATED_BY", "properties": {"efficacy": "high", "duration": "6-9 months"}},
    {"source": "d3", "target": "t2", "type": "TREATED_BY", "properties": {"efficacy": "variable"}},
    {"source": "d4", "target": "t3", "type": "TREATED_BY", "properties": {"efficacy": "high"}},
    # Treatment - Drug relationships
    {"source": "t1", "target": "dr1", "type": "USES_DRUG", "properties": {}},
    {"source": "t4", "target": "dr2", "type": "USES_DRUG", "properties": {}},
    {"source": "t3", "target": "dr3", "type": "USES_DRUG", "properties": {}},
    # Disease - Diagnostic Test relationships
    {"source": "d1", "target": "dt1", "type": "DIAGNOSED_BY", "properties": {"sensitivity": "high"}},
    {"source": "d2", "target": "dt1", "type": "DIAGNOSED_BY", "properties": {"sensitivity": "moderate"}},
    {"source": "d2", "target": "dt3", "type": "DIAGNOSED_BY", "properties": {"sensitivity": "high"}},
    {"source": "d3", "target": "dt2", "type": "DIAGNOSED_BY", "properties": {"sensitivity": "high"}},
]


class NodeData(BaseModel):
    id: str
    label: str
    type: str
    properties: Dict[str, Any] = {}


class EdgeData(BaseModel):
    source: str
    target: str
    type: str
    properties: Dict[str, Any] = {}


class GraphData(BaseModel):
    nodes: List[NodeData]
    edges: List[EdgeData]


class CypherQuery(BaseModel):
    query: str
    parameters: Dict[str, Any] = {}


@router.get("/status")
async def graph_status():
    """Check Neo4j connection status."""
    driver = get_driver()
    if driver:
        try:
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            return {
                "connected": True,
                "uri": settings.NEO4J_URI,
                "message": "Neo4j 연결됨",
                "mode": "live"
            }
        except Exception as e:
            return {
                "connected": True,
                "uri": settings.NEO4J_URI,
                "message": f"데모 모드 (Neo4j 미연결)",
                "mode": "demo"
            }
    return {
        "connected": True,
        "uri": settings.NEO4J_URI,
        "message": "데모 모드 (Neo4j 미설치)",
        "mode": "demo"
    }


def get_mock_data(node_type: Optional[str] = None, limit: int = 50):
    """Get mock graph data for demo mode."""
    if node_type:
        nodes = [n for n in MOCK_NODES if n["type"] == node_type][:limit]
    else:
        nodes = MOCK_NODES[:limit]

    node_ids = {n["id"] for n in nodes}
    edges = [e for e in MOCK_EDGES if e["source"] in node_ids and e["target"] in node_ids]

    return {
        "nodes": [NodeData(**n) for n in nodes],
        "edges": [EdgeData(**e) for e in edges]
    }


@router.get("/nodes", response_model=GraphData)
async def get_nodes(
    node_type: Optional[str] = Query(None, description="Filter by node type (Disease, Symptom, Treatment, Drug)"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of nodes")
):
    """Get nodes from the knowledge graph."""
    driver = get_driver()
    if not driver:
        # Return mock data for demo
        mock_data = get_mock_data(node_type, limit)
        return GraphData(nodes=mock_data["nodes"], edges=mock_data["edges"])

    try:
        with driver.session() as session:
            if node_type:
                query = f"""
                MATCH (n:{node_type})
                OPTIONAL MATCH (n)-[r]-(m)
                RETURN n, r, m, type(r) as rel_type
                LIMIT {limit}
                """
            else:
                query = f"""
                MATCH (n)
                OPTIONAL MATCH (n)-[r]-(m)
                RETURN n, r, m, type(r) as rel_type
                LIMIT {limit}
                """

            result = session.run(query)

            nodes_dict = {}
            edges = []

            for record in result:
                # Process source node
                n = record.get("n")
                if n:
                    node_id = str(n.element_id)
                    if node_id not in nodes_dict:
                        labels = list(n.labels)
                        nodes_dict[node_id] = NodeData(
                            id=node_id,
                            label=n.get("name", n.get("title", labels[0] if labels else "Unknown")),
                            type=labels[0] if labels else "Unknown",
                            properties=dict(n)
                        )

                # Process target node
                m = record.get("m")
                if m:
                    node_id = str(m.element_id)
                    if node_id not in nodes_dict:
                        labels = list(m.labels)
                        nodes_dict[node_id] = NodeData(
                            id=node_id,
                            label=m.get("name", m.get("title", labels[0] if labels else "Unknown")),
                            type=labels[0] if labels else "Unknown",
                            properties=dict(m)
                        )

                # Process relationship
                r = record.get("r")
                rel_type = record.get("rel_type")
                if rel_type and n and m:
                    edges.append(EdgeData(
                        source=str(n.element_id),
                        target=str(m.element_id),
                        type=rel_type,
                        properties=dict(r) if r else {}
                    ))

            return GraphData(
                nodes=list(nodes_dict.values()),
                edges=edges
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"쿼리 실행 오류: {str(e)}")


@router.get("/search")
async def search_graph(
    query: str = Query(..., min_length=1, description="Search term"),
    limit: int = Query(30, ge=1, le=100)
):
    """Search nodes by name or property."""
    driver = get_driver()
    if not driver:
        # Search in mock data
        query_lower = query.lower()
        matched_nodes = []
        for node in MOCK_NODES:
            props = node["properties"]
            if (query_lower in node["label"].lower() or
                query_lower in props.get("name", "").lower() or
                query_lower in props.get("name_en", "").lower() or
                query_lower in props.get("description", "").lower()):
                matched_nodes.append(node)

        matched_nodes = matched_nodes[:limit]
        node_ids = {n["id"] for n in matched_nodes}
        matched_edges = [e for e in MOCK_EDGES if e["source"] in node_ids or e["target"] in node_ids]

        # Add connected nodes
        for edge in matched_edges:
            for node in MOCK_NODES:
                if node["id"] == edge["source"] or node["id"] == edge["target"]:
                    if node["id"] not in node_ids:
                        matched_nodes.append(node)
                        node_ids.add(node["id"])

        return {
            "nodes": matched_nodes,
            "edges": matched_edges,
            "query": query
        }

    try:
        with driver.session() as session:
            cypher = """
            MATCH (n)
            WHERE toLower(coalesce(n.name, '')) CONTAINS toLower($search_term)
               OR toLower(coalesce(n.name_en, '')) CONTAINS toLower($search_term)
               OR toLower(coalesce(n.description, '')) CONTAINS toLower($search_term)
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN n, r, m, type(r) as rel_type
            LIMIT $max_results
            """

            result = session.run(cypher, search_term=query, max_results=limit)

            nodes_dict = {}
            edges = []

            for record in result:
                n = record.get("n")
                if n:
                    node_id = str(n.element_id)
                    if node_id not in nodes_dict:
                        labels = list(n.labels)
                        nodes_dict[node_id] = {
                            "id": node_id,
                            "label": n.get("name", n.get("title", labels[0] if labels else "Unknown")),
                            "type": labels[0] if labels else "Unknown",
                            "properties": dict(n)
                        }

                m = record.get("m")
                if m:
                    node_id = str(m.element_id)
                    if node_id not in nodes_dict:
                        labels = list(m.labels)
                        nodes_dict[node_id] = {
                            "id": node_id,
                            "label": m.get("name", m.get("title", labels[0] if labels else "Unknown")),
                            "type": labels[0] if labels else "Unknown",
                            "properties": dict(m)
                        }

                r = record.get("r")
                rel_type = record.get("rel_type")
                if rel_type and n and m:
                    edges.append({
                        "source": str(n.element_id),
                        "target": str(m.element_id),
                        "type": rel_type,
                        "properties": dict(r) if r else {}
                    })

            return {
                "nodes": list(nodes_dict.values()),
                "edges": edges,
                "query": query
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 오류: {str(e)}")


@router.post("/query")
async def execute_cypher(body: CypherQuery):
    """Execute custom Cypher query (read-only)."""
    driver = get_driver()
    if not driver:
        raise HTTPException(status_code=503, detail="Neo4j 연결 실패")

    # Block write operations
    query_upper = body.query.upper()
    if any(word in query_upper for word in ["CREATE", "DELETE", "SET", "REMOVE", "MERGE", "DROP"]):
        raise HTTPException(status_code=400, detail="읽기 전용 쿼리만 허용됩니다")

    try:
        with driver.session() as session:
            result = session.run(body.query, body.parameters)
            records = [dict(record) for record in result]

            return {
                "success": True,
                "data": records[:100],  # Limit results
                "count": len(records)
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"쿼리 오류: {str(e)}")


@router.get("/schema")
async def get_schema():
    """Get graph database schema (node labels and relationship types)."""
    driver = get_driver()
    if not driver:
        # Return mock schema
        labels = list(set(n["type"] for n in MOCK_NODES))
        rel_types = list(set(e["type"] for e in MOCK_EDGES))
        counts = {}
        for label in labels:
            counts[label] = len([n for n in MOCK_NODES if n["type"] == label])
        return {
            "labels": labels,
            "relationship_types": rel_types,
            "node_counts": counts,
            "mode": "demo"
        }

    try:
        with driver.session() as session:
            # Get node labels
            labels_result = session.run("CALL db.labels()")
            labels = [record["label"] for record in labels_result]

            # Get relationship types
            rel_result = session.run("CALL db.relationshipTypes()")
            rel_types = [record["relationshipType"] for record in rel_result]

            # Get node counts per label
            counts = {}
            for label in labels:
                count_result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                counts[label] = count_result.single()["count"]

            return {
                "labels": labels,
                "relationship_types": rel_types,
                "node_counts": counts
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"스키마 조회 오류: {str(e)}")
