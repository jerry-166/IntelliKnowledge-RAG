# IntelliKnowledge-RAG API 接口设计文档

## 1. API概览

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| **Base URL** | `https://api.intelliknowledge.com/api/v1` |
| **协议** | HTTPS |
| **数据格式** | JSON |
| **认证方式** | JWT Bearer Token |
| **API版本** | v1.0 |

### 1.2 通用响应格式

```json
// 成功响应
{
  "code": 200,
  "message": "Success",
  "data": { ... },
  "timestamp": "2025-12-04T10:30:00Z"
}

// 错误响应
{
  "code": 400,
  "message": "Invalid request",
  "error": {
    "type": "ValidationError",
    "details": [
      {"field": "email", "message": "Invalid email format"}
    ]
  },
  "timestamp": "2025-12-04T10:30:00Z"
}
```

### 1.3 状态码说明

| 状态码 | 说明 |
|--------|------|
| 200 | 请求成功 |
| 201 | 创建成功 |
| 400 | 请求参数错误 |
| 401 | 未授权 |
| 403 | 禁止访问 |
| 404 | 资源不存在 |
| 429 | 请求过于频繁 |
| 500 | 服务器内部错误 |
| 503 | 服务不可用 |

---

## 2. 认证授权

### 2.1 用户注册

```http
POST /auth/register
Content-Type: application/json

{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "SecurePass123!"
}
```

**响应示例**：
```json
{
  "code": 201,
  "message": "User registered successfully",
  "data": {
    "user_id": 1001,
    "username": "john_doe",
    "email": "john@example.com",
    "created_at": "2025-12-04T10:30:00Z"
  }
}
```

### 2.2 用户登录

```http
POST /auth/login
Content-Type: application/json

{
  "email": "john@example.com",
  "password": "SecurePass123!"
}
```

**响应示例**：
```json
{
  "code": 200,
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "Bearer",
    "expires_in": 3600,
    "user": {
      "id": 1001,
      "username": "john_doe",
      "email": "john@example.com"
    }
  }
}
```

### 2.3 刷新Token

```http
POST /auth/refresh
Authorization: Bearer {refresh_token}
```

**响应示例**：
```json
{
  "code": 200,
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "expires_in": 3600
  }
}
```

---

## 3. 知识库管理

### 3.1 创建知识库

```http
POST /collections
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "name": "技术文档库",
  "description": "存储所有技术相关文档",
  "is_public": false,
  "settings": {
    "embedding_model": "text-embedding-3-large",
    "chunk_size": 512,
    "chunk_overlap": 50
  }
}
```

**响应示例**：
```json
{
  "code": 201,
  "data": {
    "id": 10,
    "name": "技术文档库",
    "description": "存储所有技术相关文档",
    "user_id": 1001,
    "is_public": false,
    "document_count": 0,
    "created_at": "2025-12-04T10:30:00Z"
  }
}
```

### 3.2 获取知识库列表

```http
GET /collections?page=1&page_size=20&sort=created_at&order=desc
Authorization: Bearer {access_token}
```

**响应示例**：
```json
{
  "code": 200,
  "data": {
    "items": [
      {
        "id": 10,
        "name": "技术文档库",
        "description": "存储所有技术相关文档",
        "document_count": 25,
        "created_at": "2025-12-04T10:30:00Z",
        "updated_at": "2025-12-04T15:20:00Z"
      }
    ],
    "pagination": {
      "page": 1,
      "page_size": 20,
      "total_items": 3,
      "total_pages": 1
    }
  }
}
```

### 3.3 更新知识库

```http
PUT /collections/{collection_id}
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "name": "更新后的名称",
  "description": "更新后的描述",
  "settings": {
    "embedding_model": "text-embedding-3-small"
  }
}
```

### 3.4 删除知识库

```http
DELETE /collections/{collection_id}
Authorization: Bearer {access_token}
```

**响应示例**：
```json
{
  "code": 200,
  "message": "Collection deleted successfully",
  "data": {
    "deleted_documents": 25
  }
}
```

---

## 4. 文档管理

### 4.1 上传文档

```http
POST /documents/upload
Authorization: Bearer {access_token}
Content-Type: multipart/form-data

file: [binary file data]
collection_id: 10
metadata: {
  "author": "John Doe",
  "tags": ["AI", "RAG", "LangChain"],
  "source": "official_docs"
}
```

**响应示例**：
```json
{
  "code": 201,
  "message": "Document uploaded successfully",
  "data": {
    "document_id": 1001,
    "file_name": "langchain_tutorial.pdf",
    "file_type": "pdf",
    "file_size": 2048576,
    "status": "processing",
    "collection_id": 10,
    "created_at": "2025-12-04T10:30:00Z",
    "estimated_completion": "2025-12-04T10:31:00Z"
  }
}
```

### 4.2 批量上传文档

```http
POST /documents/batch-upload
Authorization: Bearer {access_token}
Content-Type: multipart/form-data

files: [file1, file2, file3, ...]
collection_id: 10
```

**响应示例**：
```json
{
  "code": 201,
  "data": {
    "batch_id": "batch_20251204_001",
    "total_files": 10,
    "uploaded": 10,
    "failed": 0,
    "documents": [
      {
        "document_id": 1001,
        "file_name": "file1.pdf",
        "status": "processing"
      }
    ]
  }
}
```

### 4.3 获取文档处理状态

```http
GET /documents/{document_id}/status
Authorization: Bearer {access_token}
```

**响应示例**：
```json
{
  "code": 200,
  "data": {
    "document_id": 1001,
    "status": "completed",  // processing, completed, failed
    "progress": 100,
    "chunks_created": 45,
    "embeddings_created": 45,
    "knowledge_graph_built": true,
    "processing_time": 15.5,  // seconds
    "error": null
  }
}
```

### 4.4 获取文档列表

```http
GET /documents?collection_id=10&page=1&page_size=20&file_type=pdf&sort=created_at
Authorization: Bearer {access_token}
```

**Query参数**：
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| collection_id | int | 否 | 知识库ID |
| page | int | 否 | 页码，默认1 |
| page_size | int | 否 | 每页数量，默认20 |
| file_type | string | 否 | 文件类型过滤 |
| search | string | 否 | 搜索关键词 |
| sort | string | 否 | 排序字段 |
| order | string | 否 | asc/desc |

**响应示例**：
```json
{
  "code": 200,
  "data": {
    "items": [
      {
        "id": 1001,
        "title": "LangChain官方教程",
        "file_name": "langchain_tutorial.pdf",
        "file_type": "pdf",
        "file_size": 2048576,
        "chunk_count": 45,
        "collection_id": 10,
        "metadata": {
          "author": "John Doe",
          "tags": ["AI", "RAG"]
        },
        "created_at": "2025-12-04T10:30:00Z"
      }
    ],
    "pagination": {
      "page": 1,
      "page_size": 20,
      "total_items": 25,
      "total_pages": 2
    }
  }
}
```

### 4.5 获取文档详情

```http
GET /documents/{document_id}
Authorization: Bearer {access_token}
```

**响应示例**：
```json
{
  "code": 200,
  "data": {
    "id": 1001,
    "title": "LangChain官方教程",
    "file_name": "langchain_tutorial.pdf",
    "file_type": "pdf",
    "file_size": 2048576,
    "file_hash": "sha256:a1b2c3...",
    "collection": {
      "id": 10,
      "name": "技术文档库"
    },
    "chunks": [
      {
        "id": 50001,
        "text": "LangChain是一个用于构建LLM应用的框架...",
        "page_number": 1,
        "chunk_index": 0,
        "token_count": 128
      }
    ],
    "metadata": {
      "author": "John Doe",
      "tags": ["AI", "RAG"],
      "page_count": 50,
      "language": "zh-CN"
    },
    "statistics": {
      "total_chunks": 45,
      "total_tokens": 5760,
      "entities_extracted": 25
    },
    "created_at": "2025-12-04T10:30:00Z",
    "updated_at": "2025-12-04T10:31:00Z"
  }
}
```

### 4.6 更新文档元数据

```http
PATCH /documents/{document_id}
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "title": "更新后的标题",
  "metadata": {
    "author": "Jane Smith",
    "tags": ["AI", "RAG", "LangGraph"]
  }
}
```

### 4.7 删除文档

```http
DELETE /documents/{document_id}
Authorization: Bearer {access_token}
```

**响应示例**：
```json
{
  "code": 200,
  "message": "Document deleted successfully",
  "data": {
    "deleted_chunks": 45,
    "deleted_embeddings": 45
  }
}
```

---

## 5. 搜索与检索

### 5.1 向量检索

```http
POST /search/vector
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "query": "什么是RAG？",
  "collection_id": 10,
  "top_k": 5,
  "filters": {
    "file_type": "pdf",
    "tags": ["AI"],
    "date_range": {
      "start": "2024-01-01",
      "end": "2025-12-31"
    }
  },
  "rerank": true
}
```

**响应示例**：
```json
{
  "code": 200,
  "data": {
    "query": "什么是RAG？",
    "results": [
      {
        "chunk_id": 50001,
        "document_id": 1001,
        "document_title": "LangChain官方教程",
        "text": "RAG（Retrieval-Augmented Generation）是一种将检索与生成相结合的技术...",
        "score": 0.92,
        "page_number": 5,
        "metadata": {
          "file_type": "pdf",
          "author": "John Doe"
        }
      }
    ],
    "search_metadata": {
      "total_results": 15,
      "search_time_ms": 35,
      "reranked": true
    }
  }
}
```

### 5.2 混合检索

```http
POST /search/hybrid
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "query": "如何使用LangGraph构建Agent？",
  "collection_id": 10,
  "top_k": 10,
  "strategy": "hybrid",  // vector, keyword, hybrid, graph
  "weights": {
    "vector": 0.6,
    "keyword": 0.3,
    "graph": 0.1
  },
  "rerank": true
}
```

### 5.3 关键词检索

```http
POST /search/keyword
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "query": "LangGraph Agent workflow",
  "collection_id": 10,
  "top_k": 10,
  "algorithm": "bm25"  // bm25, tfidf
}
```

---

## 6. 智能问答 (Agentic RAG)

### 6.1 同步问答

```http
POST /chat
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "query": "请解释LangGraph的工作原理",
  "collection_id": 10,
  "conversation_id": 5001,  // 可选，用于多轮对话
  "stream": false,
  "config": {
    "model": "gpt-4-turbo",
    "temperature": 0.7,
    "max_tokens": 2000,
    "enable_reflection": true,  // 启用自我反思
    "max_iterations": 3
  }
}
```

**响应示例**：
```json
{
  "code": 200,
  "data": {
    "message_id": 9001,
    "conversation_id": 5001,
    "answer": "LangGraph是一个用于构建有状态、多参与者应用的框架...",
    "sources": [
      {
        "document_id": 1001,
        "document_title": "LangChain官方教程",
        "chunk_id": 50001,
        "text": "...",
        "page_number": 10,
        "relevance_score": 0.92
      }
    ],
    "metadata": {
      "model": "gpt-4-turbo",
      "tokens_used": 450,
      "confidence": 0.88,
      "iterations": 1,
      "search_time_ms": 35,
      "generation_time_ms": 1200
    },
    "created_at": "2025-12-04T10:35:00Z"
  }
}
```

### 6.2 流式问答 (SSE)

```http
POST /chat/stream
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "query": "请详细介绍RAG的实现流程",
  "collection_id": 10,
  "conversation_id": 5001,
  "stream": true
}
```

**响应示例** (Server-Sent Events):
```
data: {"type": "start", "message_id": 9002}

data: {"type": "planning", "content": "分解查询为3个子问题..."}

data: {"type": "retrieval", "content": "检索到15个相关文档..."}

data: {"type": "token", "content": "RAG"}

data: {"type": "token", "content": "的实现"}

data: {"type": "token", "content": "流程"}

data: {"type": "source", "source": {"document_id": 1001, "chunk_id": 50001}}

data: {"type": "end", "metadata": {"tokens_used": 450, "confidence": 0.88}}
```

### 6.3 获取对话历史

```http
GET /conversations/{conversation_id}/messages?page=1&page_size=50
Authorization: Bearer {access_token}
```

**响应示例**：
```json
{
  "code": 200,
  "data": {
    "conversation_id": 5001,
    "title": "关于RAG的讨论",
    "messages": [
      {
        "id": 9001,
        "role": "user",
        "content": "什么是RAG？",
        "created_at": "2025-12-04T10:30:00Z"
      },
      {
        "id": 9002,
        "role": "assistant",
        "content": "RAG（Retrieval-Augmented Generation）是...",
        "sources": [...],
        "metadata": {
          "tokens_used": 450,
          "confidence": 0.88
        },
        "created_at": "2025-12-04T10:30:15Z"
      }
    ],
    "pagination": {
      "page": 1,
      "page_size": 50,
      "total_messages": 10
    }
  }
}
```

### 6.4 创建新对话

```http
POST /conversations
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "title": "技术讨论",
  "collection_id": 10
}
```

### 6.5 删除对话

```http
DELETE /conversations/{conversation_id}
Authorization: Bearer {access_token}
```

---

## 7. 知识图谱

### 7.1 查询实体

```http
GET /graph/entities?name=LangChain&type=Technology
Authorization: Bearer {access_token}
```

**响应示例**：
```json
{
  "code": 200,
  "data": {
    "entities": [
      {
        "id": "entity_001",
        "name": "LangChain",
        "type": "Technology",
        "description": "一个用于构建LLM应用的框架",
        "properties": {
          "category": "AI Framework",
          "release_year": "2022"
        },
        "mention_count": 15,
        "documents": [1001, 1002, 1003]
      }
    ]
  }
}
```

### 7.2 查询实体关系

```http
GET /graph/entities/{entity_id}/relations?max_depth=2
Authorization: Bearer {access_token}
```

**响应示例**：
```json
{
  "code": 200,
  "data": {
    "entity": {
      "id": "entity_001",
      "name": "LangChain"
    },
    "relations": [
      {
        "source": "LangChain",
        "relation": "PART_OF",
        "target": "LangGraph",
        "weight": 0.9,
        "source_documents": [1001]
      },
      {
        "source": "LangChain",
        "relation": "USED_FOR",
        "target": "RAG",
        "weight": 0.95,
        "source_documents": [1001, 1002]
      }
    ]
  }
}
```

### 7.3 查询子图

```http
POST /graph/subgraph
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "entities": ["LangChain", "RAG", "LangGraph"],
  "max_hops": 2,
  "relation_types": ["RELATED_TO", "PART_OF", "USED_FOR"]
}
```

**响应示例**：
```json
{
  "code": 200,
  "data": {
    "nodes": [
      {
        "id": "entity_001",
        "name": "LangChain",
        "type": "Technology"
      },
      {
        "id": "entity_002",
        "name": "RAG",
        "type": "Concept"
      }
    ],
    "edges": [
      {
        "source": "entity_001",
        "target": "entity_002",
        "relation": "USED_FOR",
        "weight": 0.95
      }
    ],
    "metadata": {
      "node_count": 5,
      "edge_count": 8
    }
  }
}
```

### 7.4 图谱可视化数据

```http
GET /graph/visualize?collection_id=10&layout=force
Authorization: Bearer {access_token}
```

**Query参数**：
- `layout`: force, hierarchical, circular
- `max_nodes`: 最大节点数
- `min_weight`: 最小边权重

---

## 8. 系统配置

### 8.1 获取配置

```http
GET /settings
Authorization: Bearer {access_token}
```

**响应示例**：
```json
{
  "code": 200,
  "data": {
    "llm": {
      "provider": "openai",
      "model": "gpt-4-turbo",
      "temperature": 0.7,
      "max_tokens": 2000
    },
    "embedding": {
      "model": "text-embedding-3-large",
      "dimension": 1536,
      "batch_size": 32
    },
    "retrieval": {
      "top_k": 5,
      "rerank": true,
      "hybrid_weight": {
        "vector": 0.6,
        "keyword": 0.3,
        "graph": 0.1
      }
    },
    "agentic": {
      "enable_reflection": true,
      "max_iterations": 3,
      "confidence_threshold": 0.8
    }
  }
}
```

### 8.2 更新配置

```http
PUT /settings
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "llm": {
    "model": "gpt-4-turbo",
    "temperature": 0.8
  },
  "retrieval": {
    "top_k": 10
  }
}
```

### 8.3 配置API密钥

```http
POST /settings/api-keys
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "provider": "openai",
  "api_key": "sk-...",
  "base_url": "https://api.openai.com/v1"
}
```

**响应示例**：
```json
{
  "code": 200,
  "message": "API key configured successfully",
  "data": {
    "provider": "openai",
    "api_key_preview": "sk-...xyz",
    "status": "valid"
  }
}
```

### 8.4 测试API连接

```http
POST /settings/test-connection
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "provider": "openai",
  "model": "gpt-4-turbo"
}
```

**响应示例**：
```json
{
  "code": 200,
  "data": {
    "status": "success",
    "latency_ms": 150,
    "model_info": {
      "name": "gpt-4-turbo",
      "context_window": 128000,
      "max_output_tokens": 4096
    }
  }
}
```

---

## 9. 统计与分析

### 9.1 获取系统统计

```http
GET /statistics/overview
Authorization: Bearer {access_token}
```

**响应示例**：
```json
{
  "code": 200,
  "data": {
    "user": {
      "total_collections": 3,
      "total_documents": 50,
      "total_chunks": 2250,
      "storage_used_mb": 125.5
    },
    "usage": {
      "total_queries": 150,
      "total_tokens": 75000,
      "estimated_cost_usd": 0.15
    },
    "period": "last_30_days"
  }
}
```

### 9.2 获取知识库统计

```http
GET /statistics/collections/{collection_id}
Authorization: Bearer {access_token}
```

**响应示例**：
```json
{
  "code": 200,
  "data": {
    "collection_id": 10,
    "document_stats": {
      "total_documents": 25,
      "by_type": {
        "pdf": 15,
        "markdown": 8,
        "docx": 2
      }
    },
    "query_stats": {
      "total_queries": 80,
      "avg_response_time_ms": 1200,
      "avg_confidence": 0.85
    },
    "popular_queries": [
      {"query": "什么是RAG", "count": 10},
      {"query": "如何使用LangGraph", "count": 8}
    ]
  }
}
```

### 9.3 获取使用趋势

```http
GET /statistics/trends?period=7d&metric=queries
Authorization: Bearer {access_token}
```

**Query参数**：
- `period`: 7d, 30d, 90d
- `metric`: queries, tokens, documents

**响应示例**：
```json
{
  "code": 200,
  "data": {
    "metric": "queries",
    "period": "7d",
    "data_points": [
      {"date": "2025-11-28", "value": 15},
      {"date": "2025-11-29", "value": 20},
      {"date": "2025-11-30", "value": 18}
    ]
  }
}
```

---

## 10. 管理员接口

### 10.1 获取所有用户

```http
GET /admin/users?page=1&page_size=20
Authorization: Bearer {admin_access_token}
```

### 10.2 系统健康检查

```http
GET /admin/health
```

**响应示例**：
```json
{
  "code": 200,
  "data": {
    "status": "healthy",
    "components": {
      "database": {
        "status": "up",
        "latency_ms": 2
      },
      "vector_db": {
        "status": "up",
        "latency_ms": 5
      },
      "redis": {
        "status": "up",
        "latency_ms": 1
      },
      "llm_api": {
        "status": "up",
        "latency_ms": 150
      }
    },
    "uptime_seconds": 86400
  }
}
```

### 10.3 获取系统指标

```http
GET /admin/metrics
Authorization: Bearer {admin_access_token}
```

**响应示例**：
```json
{
  "code": 200,
  "data": {
    "requests": {
      "total": 10000,
      "success": 9850,
      "error": 150,
      "avg_latency_ms": 350
    },
    "resources": {
      "cpu_usage_percent": 45.2,
      "memory_usage_mb": 2048,
      "disk_usage_gb": 50.5
    },
    "cache": {
      "hit_rate": 0.85,
      "total_keys": 5000
    }
  }
}
```

---

## 11. Webhook配置

### 11.1 创建Webhook

```http
POST /webhooks
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "url": "https://example.com/webhook",
  "events": ["document.processed", "document.failed"],
  "secret": "webhook_secret_key"
}
```

### 11.2 Webhook事件示例

```json
// document.processed事件
{
  "event": "document.processed",
  "timestamp": "2025-12-04T10:30:00Z",
  "data": {
    "document_id": 1001,
    "collection_id": 10,
    "status": "completed",
    "chunks_created": 45,
    "processing_time": 15.5
  }
}

// document.failed事件
{
  "event": "document.failed",
  "timestamp": "2025-12-04T10:30:00Z",
  "data": {
    "document_id": 1002,
    "collection_id": 10,
    "status": "failed",
    "error": "Unsupported file format"
  }
}
```

---

## 12. 错误码说明

| 错误码 | 说明 | 解决方案 |
|--------|------|----------|
| E1001 | Invalid authentication token | 重新登录获取token |
| E1002 | Token expired | 使用refresh_token刷新 |
| E2001 | Collection not found | 检查collection_id |
| E2002 | Document not found | 检查document_id |
| E3001 | File size exceeds limit | 压缩文件或分割上传 |
| E3002 | Unsupported file type | 查看支持的文件类型列表 |
| E4001 | Query too long | 缩短查询内容 |
| E4002 | Rate limit exceeded | 降低请求频率 |
| E5001 | LLM API error | 检查API配置和余额 |
| E5002 | Vector DB connection failed | 联系管理员 |

---

## 13. 速率限制

| 端点 | 限制 | 窗口期 |
|------|------|--------|
| `/auth/login` | 5次 | 1分钟 |
| `/documents/upload` | 20次 | 1分钟 |
| `/chat` | 60次 | 1分钟 |
| `/search/*` | 100次 | 1分钟 |
| 其他端点 | 300次 | 1分钟 |

**响应头**：
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1638360000
```

---

## 14. SDK示例

### Python SDK

```python
from intelliknowledge import Client

# 初始化客户端
client = Client(
    api_key="your_api_key",
    base_url="https://api.intelliknowledge.com/api/v1"
)

# 上传文档
document = client.documents.upload(
    file_path="tutorial.pdf",
    collection_id=10
)

# 等待处理完成
document.wait_for_completion()

# 问答
response = client.chat.create(
    query="什么是RAG？",
    collection_id=10,
    stream=False
)

print(response.answer)
print(response.sources)

# 流式问答
for chunk in client.chat.stream(
    query="详细解释RAG的工作原理",
    collection_id=10
):
    print(chunk, end="")
```

### JavaScript SDK

```javascript
import { IntelliKnowledge } from '@intelliknowledge/sdk';

const client = new IntelliKnowledge({
  apiKey: 'your_api_key',
  baseUrl: 'https://api.intelliknowledge.com/api/v1'
});

// 上传文档
const document = await client.documents.upload({
  file: fileBlob,
  collectionId: 10
});

// 问答
const response = await client.chat.create({
  query: '什么是RAG？',
  collectionId: 10,
  stream: false
});

console.log(response.answer);

// 流式问答
const stream = await client.chat.stream({
  query: '详细解释RAG的工作原理',
  collectionId: 10
});

for await (const chunk of stream) {
  process.stdout.write(chunk);
}
```

---

## 15. Postman Collection

提供完整的Postman Collection，包含所有API端点的示例请求。

**下载链接**：`https://api.intelliknowledge.com/postman/collection.json`

---

**文档版本**：v1.0
**最后更新**：2025-12-04
**联系方式**：api-support@intelliknowledge.com
