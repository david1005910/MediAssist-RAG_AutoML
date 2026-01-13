"""Neo4j Graph Database router for medical knowledge graph."""
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from neo4j import GraphDatabase

from ..config import settings

router = APIRouter(prefix="/api/v1/graph", tags=["Knowledge Graph"])

# Neo4j driver
_driver = None


def get_driver():
    """Get or create Neo4j driver."""
    global _driver
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
                "message": "Neo4j 연결됨"
            }
        except Exception as e:
            return {
                "connected": False,
                "uri": settings.NEO4J_URI,
                "message": f"연결 오류: {str(e)}"
            }
    return {
        "connected": False,
        "uri": settings.NEO4J_URI,
        "message": "Neo4j 드라이버 초기화 실패"
    }


@router.get("/nodes", response_model=GraphData)
async def get_nodes(
    node_type: Optional[str] = Query(None, description="Filter by node type (Disease, Symptom, Treatment, Drug)"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of nodes")
):
    """Get nodes from the knowledge graph."""
    driver = get_driver()
    if not driver:
        raise HTTPException(status_code=503, detail="Neo4j 연결 실패")

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
        raise HTTPException(status_code=503, detail="Neo4j 연결 실패")

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
        raise HTTPException(status_code=503, detail="Neo4j 연결 실패")

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
