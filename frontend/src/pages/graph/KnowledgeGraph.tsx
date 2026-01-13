import { useState, useEffect, useRef, useCallback } from 'react'
import { Link } from 'react-router-dom'
import ForceGraph3D from 'react-force-graph-3d'
import * as THREE from 'three'

interface GraphNode {
  id: string
  label: string
  type: string
  properties: Record<string, unknown>
}

interface GraphEdge {
  source: string
  target: string
  type: string
  properties: Record<string, unknown>
}

interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

interface SchemaData {
  labels: string[]
  relationship_types: string[]
  node_counts: Record<string, number>
}

// Neo4j Knowledge Graph colors
const NEO4J_COLORS: Record<string, string> = {
  Disease: '#ef4444',
  Symptom: '#f59e0b',
  Treatment: '#10b981',
  Drug: '#3b82f6',
  DiagnosticTest: '#8b5cf6',
  ImageFinding: '#ec4899',
  RiskLevel: '#f97316',
  Default: '#6b7280',
}

const NEO4J_SIZES: Record<string, number> = {
  Disease: 8,
  Symptom: 5,
  Treatment: 6,
  Drug: 5,
  DiagnosticTest: 5,
  ImageFinding: 4,
  RiskLevel: 4,
  Default: 4,
}

export default function KnowledgeGraph() {
  // Neo4j Graph State
  const [neo4jData, setNeo4jData] = useState<{ nodes: any[]; links: any[] }>({ nodes: [], links: [] })
  const [schema, setSchema] = useState<SchemaData | null>(null)
  const [neo4jStatus, setNeo4jStatus] = useState<{ connected: boolean; message: string } | null>(null)
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)

  // Search State
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedType, setSelectedType] = useState<string>('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // JSON Viewer State
  const [selectedJsonData, setSelectedJsonData] = useState<object | null>(null)
  const [jsonCopied, setJsonCopied] = useState(false)

  const graphRef = useRef<any>()

  // Check connection status on mount
  useEffect(() => {
    checkNeo4jStatus()
    loadNeo4jSchema()
  }, [])

  const checkNeo4jStatus = async () => {
    try {
      const response = await fetch('http://localhost:8003/api/v1/graph/status')
      const data = await response.json()
      setNeo4jStatus(data)
      if (data.connected) {
        loadNeo4jGraph()
      }
    } catch (err) {
      setNeo4jStatus({ connected: false, message: 'API 연결 실패' })
    }
  }

  const loadNeo4jSchema = async () => {
    try {
      const response = await fetch('http://localhost:8003/api/v1/graph/schema')
      if (response.ok) {
        const data = await response.json()
        setSchema(data)
      }
    } catch (err) {
      console.error('Schema load error:', err)
    }
  }

  const loadNeo4jGraph = async (nodeType?: string) => {
    setIsLoading(true)
    setError(null)
    try {
      const url = nodeType
        ? `http://localhost:8003/api/v1/graph/nodes?node_type=${nodeType}&limit=100`
        : 'http://localhost:8003/api/v1/graph/nodes?limit=100'

      const response = await fetch(url)
      if (!response.ok) throw new Error('그래프 로드 실패')

      const data: GraphData = await response.json()
      transformNeo4jData(data)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const searchGraph = async () => {
    if (!searchQuery.trim()) {
      loadNeo4jGraph(selectedType || undefined)
      return
    }

    setIsLoading(true)
    setError(null)
    try {
      const response = await fetch(
        `http://localhost:8003/api/v1/graph/search?query=${encodeURIComponent(searchQuery)}&limit=50`
      )

      if (response.ok) {
        const data: GraphData = await response.json()
        transformNeo4jData(data)
      }
    } catch (err: any) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const transformNeo4jData = (data: GraphData) => {
    const nodes = data.nodes.map((node) => ({
      id: node.id,
      name: node.label,
      type: node.type,
      properties: node.properties,
      color: NEO4J_COLORS[node.type] || NEO4J_COLORS.Default,
      size: NEO4J_SIZES[node.type] || NEO4J_SIZES.Default,
    }))

    const nodeIds = new Set(nodes.map((n) => n.id))
    const links = data.edges
      .filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target))
      .map((edge) => ({
        source: edge.source,
        target: edge.target,
        type: edge.type,
        properties: edge.properties,
      }))

    setNeo4jData({ nodes, links })
  }

  const handleNodeClick = useCallback((node: any) => {
    const nodeData = {
      id: node.id,
      label: node.name,
      type: node.type,
      properties: node.properties,
    }
    setSelectedNode(nodeData)
    setSelectedJsonData(nodeData)

    if (graphRef.current) {
      const distance = 120
      const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z)
      graphRef.current.cameraPosition(
        { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio },
        node,
        2000
      )
    }
  }, [])

  // JSON viewer helper functions
  const copyJsonToClipboard = async () => {
    if (selectedJsonData) {
      await navigator.clipboard.writeText(JSON.stringify(selectedJsonData, null, 2))
      setJsonCopied(true)
      setTimeout(() => setJsonCopied(false), 2000)
    }
  }

  const showAllGraphJson = () => {
    setSelectedJsonData({
      nodes: neo4jData.nodes,
      links: neo4jData.links,
      stats: {
        nodeCount: neo4jData.nodes.length,
        linkCount: neo4jData.links.length,
      }
    })
  }

  const handleLinkClick = (link: any) => {
    setSelectedJsonData({
      type: 'relationship',
      source: link.source?.id || link.source,
      target: link.target?.id || link.target,
      relationshipType: link.type,
      properties: link.properties,
    })
  }

  const handleTypeFilter = (type: string) => {
    setSelectedType(type)
    loadNeo4jGraph(type || undefined)
  }

  // Create 3D node objects
  const nodeThreeObject = useCallback((node: any) => {
    const size = node.size || 4
    const geometry = new THREE.SphereGeometry(size, 16, 16)
    const material = new THREE.MeshLambertMaterial({
      color: node.color,
      transparent: true,
      opacity: 0.9,
    })
    const sphere = new THREE.Mesh(geometry, material)

    const sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: createTextTexture(node.name),
        transparent: true,
      })
    )
    sprite.scale.set(40, 20, 1)
    sprite.position.set(0, size + 8, 0)

    const group = new THREE.Group()
    group.add(sphere)
    group.add(sprite)
    return group
  }, [])

  // Create text texture for labels
  const createTextTexture = (text: string, color: string = '#ffffff') => {
    const canvas = document.createElement('canvas')
    const context = canvas.getContext('2d')!
    canvas.width = 256
    canvas.height = 128

    context.fillStyle = 'rgba(0, 0, 0, 0)'
    context.fillRect(0, 0, canvas.width, canvas.height)

    context.font = 'bold 22px Arial'
    context.fillStyle = color
    context.textAlign = 'center'
    context.textBaseline = 'middle'
    context.fillText(text.substring(0, 20), canvas.width / 2, canvas.height / 2)

    const texture = new THREE.CanvasTexture(canvas)
    texture.needsUpdate = true
    return texture
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#1A1D21] to-[#15171A]">
      <header className="metal-header">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center gap-4">
          <Link to="/dashboard" className="text-metal-text-muted hover:text-metal-text-light transition-colors">
            ← 대시보드
          </Link>
          <h1 className="text-xl font-bold text-metal-text-light">의료 지식 그래프</h1>
          <div className="flex items-center gap-3 ml-auto">
            {neo4jStatus && (
              <span className={`flex items-center gap-2 text-xs px-2 py-1 rounded-metal-sm ${neo4jStatus.connected ? 'metal-badge-green' : 'metal-badge-red'}`}>
                <span className={`w-2 h-2 rounded-full ${neo4jStatus.connected ? 'metal-status-online' : 'metal-status-offline'}`}></span>
                Neo4j: {neo4jStatus.message}
              </span>
            )}
          </div>
        </div>
      </header>

      <main className="flex h-[calc(100vh-64px)]">
        {/* Left Sidebar */}
        <div className="w-64 p-4 overflow-y-auto flex-shrink-0 metal-sidebar">
          {/* Search */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-metal-text-mid mb-2">검색</label>
            <div className="flex gap-2">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && searchGraph()}
                placeholder="질환, 증상, 치료법..."
                className="flex-1 px-3 py-2 metal-input text-sm"
              />
              <button
                onClick={searchGraph}
                disabled={isLoading}
                className="px-3 py-2 metal-btn disabled:opacity-50 text-sm"
              >
                검색
              </button>
            </div>
          </div>

          {/* Node Type Filter */}
          {schema && schema.labels.length > 0 && (
            <div className="mb-6">
              <label className="block text-sm font-medium text-metal-text-mid mb-2">노드 타입</label>
              <div className="flex flex-wrap gap-1">
                <button
                  onClick={() => handleTypeFilter('')}
                  className={`px-2 py-1 rounded-metal-sm text-xs transition-all ${
                    selectedType === '' ? 'metal-btn' : 'metal-btn-secondary'
                  }`}
                >
                  전체
                </button>
                {schema.labels.map((label) => (
                  <button
                    key={label}
                    onClick={() => handleTypeFilter(label)}
                    className={`px-2 py-1 rounded-metal-sm text-xs flex items-center gap-1 transition-all ${
                      selectedType === label ? 'metal-btn' : 'metal-btn-secondary'
                    }`}
                  >
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: NEO4J_COLORS[label] || NEO4J_COLORS.Default }}
                    ></span>
                    {label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Legend */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-metal-text-mid mb-2">범례</label>
            <div className="grid grid-cols-2 gap-1 text-xs">
              {Object.entries(NEO4J_COLORS).filter(([k]) => k !== 'Default').map(([type, color]) => (
                <div key={type} className="flex items-center gap-1 text-metal-text-muted">
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }}></span>
                  {type}
                </div>
              ))}
            </div>
          </div>

          {/* Selected Node Info */}
          {selectedNode && (
            <div className="mb-4 rounded-metal p-3"
              style={{
                background: 'linear-gradient(180deg, #2A2F37 0%, #252930 100%)',
                borderTop: '1px solid rgba(255,255,255,0.08)'
              }}>
              <h3 className="text-sm font-semibold text-metal-text-light mb-1">{selectedNode.label}</h3>
              <span
                className="inline-block px-2 py-0.5 rounded-metal-sm text-xs text-white mb-2"
                style={{ backgroundColor: NEO4J_COLORS[selectedNode.type] || NEO4J_COLORS.Default }}
              >
                {selectedNode.type}
              </span>
              {Object.entries(selectedNode.properties).slice(0, 5).map(([key, value]) => (
                <div key={key} className="text-xs">
                  <span className="text-metal-text-muted">{key}:</span>{' '}
                  <span className="text-metal-text-mid">{String(value).substring(0, 60)}</span>
                </div>
              ))}
            </div>
          )}

          {/* Stats */}
          <div className="text-xs text-metal-text-muted space-y-1 p-3 rounded-metal"
            style={{ background: 'rgba(0,0,0,0.2)' }}>
            <p>노드: {neo4jData.nodes.length}개</p>
            <p>관계: {neo4jData.links.length}개</p>
          </div>
        </div>

        {/* Graph Canvas Area */}
        <div className="flex-1 relative">
          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center z-10" style={{ background: 'rgba(26, 29, 33, 0.7)' }}>
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent-cyan"></div>
            </div>
          )}

          {error && (
            <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-10 px-4 py-2 rounded-metal"
              style={{
                background: 'linear-gradient(180deg, #5C2A2A 0%, #4A2222 100%)',
                borderTop: '1px solid rgba(255,255,255,0.1)'
              }}>
              <span className="text-red-300">{error}</span>
            </div>
          )}

          {/* Title Bar */}
          <div className="absolute top-0 left-0 right-0 z-10 px-4 py-2"
            style={{
              background: 'linear-gradient(90deg, rgba(230, 126, 34, 0.9) 0%, rgba(155, 89, 182, 0.9) 100%)',
              borderBottom: '1px solid rgba(255,255,255,0.1)'
            }}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="text-xl">🕸️</span>
                <div>
                  <h2 className="text-white font-bold text-base">Neo4j 의료 지식 그래프</h2>
                  <p className="text-white/70 text-xs">질병-증상-치료법 관계 네트워크</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="px-2 py-1 rounded-metal-sm text-xs" style={{ background: 'rgba(255,255,255,0.2)', color: '#FFF' }}>
                  {neo4jData.nodes.length} 노드
                </span>
                <span className="px-2 py-1 rounded-metal-sm text-xs" style={{ background: 'rgba(255,255,255,0.2)', color: '#FFF' }}>
                  {neo4jData.links.length} 관계
                </span>
              </div>
            </div>
          </div>

          {!neo4jStatus?.connected ? (
            <div className="flex items-center justify-center h-full text-metal-text-muted pt-12">
              <div className="text-center">
                <p className="text-2xl mb-2 opacity-50">🔌</p>
                <p>Neo4j 연결 필요</p>
                <button
                  onClick={checkNeo4jStatus}
                  className="mt-2 px-3 py-1 metal-btn text-sm"
                >
                  다시 연결
                </button>
              </div>
            </div>
          ) : neo4jData.nodes.length === 0 ? (
            <div className="flex items-center justify-center h-full text-metal-text-muted pt-12">
              <p>검색하거나 노드 타입을 선택하세요</p>
            </div>
          ) : (
            <div className="pt-12 h-full">
              <ForceGraph3D
                ref={graphRef}
                graphData={neo4jData}
                nodeLabel="name"
                nodeThreeObject={nodeThreeObject}
                nodeThreeObjectExtend={false}
                linkColor={() => '#4b5563'}
                linkWidth={1}
                linkOpacity={0.4}
                linkDirectionalParticles={2}
                linkDirectionalParticleWidth={2}
                linkDirectionalParticleColor={() => '#4FC3F7'}
                onNodeClick={handleNodeClick}
                onLinkClick={handleLinkClick}
                backgroundColor="#15171A"
                showNavInfo={false}
              />
            </div>
          )}
        </div>

        {/* Right Panel - JSON Viewer */}
        <div className="w-80 flex-shrink-0 flex flex-col metal-sidebar" style={{ borderLeft: '1px solid rgba(255,255,255,0.05)' }}>
          <div className="flex items-center justify-between p-3 border-b border-white/5">
            <h3 className="font-semibold text-metal-text-light text-sm">JSON 메타데이터</h3>
            <div className="flex gap-1">
              <button
                onClick={showAllGraphJson}
                disabled={neo4jData.nodes.length === 0}
                className="px-2 py-1 text-xs metal-btn disabled:opacity-50"
              >
                전체
              </button>
              <button
                onClick={copyJsonToClipboard}
                disabled={!selectedJsonData}
                className="px-2 py-1 text-xs metal-btn-secondary disabled:opacity-50"
              >
                {jsonCopied ? '복사됨!' : '복사'}
              </button>
              <button
                onClick={() => setSelectedJsonData(null)}
                disabled={!selectedJsonData}
                className="px-2 py-1 text-xs metal-btn-danger disabled:opacity-50"
              >
                지우기
              </button>
            </div>
          </div>
          <div className="flex-1 p-3 overflow-auto">
            {selectedJsonData ? (
              <pre className="text-xs p-3 rounded-metal overflow-auto max-h-full font-mono whitespace-pre-wrap metal-json-viewer">
                {JSON.stringify(selectedJsonData, null, 2)}
              </pre>
            ) : (
              <div className="text-center text-metal-text-muted py-12">
                <div className="text-3xl mb-2 opacity-30">{ }</div>
                <p className="text-sm">노드나 관계를 클릭하면</p>
                <p className="text-sm">JSON 형식으로 볼 수 있습니다</p>
              </div>
            )}
          </div>
          {selectedJsonData && (
            <div className="px-3 pb-3 text-xs text-metal-text-muted border-t border-white/5 pt-2">
              <div className="flex justify-between">
                <span>데이터 크기:</span>
                <span>{JSON.stringify(selectedJsonData).length.toLocaleString()} bytes</span>
              </div>
              <div className="flex justify-between mt-1">
                <span>키 개수:</span>
                <span>{Object.keys(selectedJsonData).length}개</span>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
