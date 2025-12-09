# Java + Python 混合架构方案

## 1. 架构概述

### 1.1 为什么选择混合架构？

本项目采用 **Java + Python 微服务混合架构**，充分发挥两种语言的优势：

| 服务类型 | 技术选型 | 理由 |
|---------|---------|------|
| **业务服务** | Java (Spring Boot) | 企业级特性、类型安全、高性能 |
| **AI服务** | Python (FastAPI) | AI生态丰富、开发效率高 |

### 1.2 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React)                      │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTPS
┌────────────────────────▼────────────────────────────────────┐
│              API Gateway (Spring Cloud Gateway)              │
│         路由 + 鉴权 + 限流 + 负载均衡 + 熔断降级                │
└────┬─────────────────┬──────────────────┬──────────────────┘
     │                 │                  │
┌────▼────────┐  ┌─────▼──────┐  ┌──────▼──────────────┐
│ Java服务集群 │  │ Java服务集群 │  │  Python服务集群      │
├─────────────┤  ├────────────┤  ├────────────────────┤
│user-service │  │collection  │  │ ingestion-service  │
│用户&权限管理  │  │-service    │  │ 文档解析&向量化      │
│             │  │知识库管理    │  │                    │
├─────────────┤  ├────────────┤  ├────────────────────┤
│document     │  │audit       │  │ rag-service        │
│-service     │  │-service    │  │ Agentic RAG引擎    │
│文档元数据管理 │  │审计日志     │  │                    │
├─────────────┤  ├────────────┤  ├────────────────────┤
│gateway      │  │statistics  │  │ graph-service      │
│-service     │  │-service    │  │ 知识图谱构建        │
│API网关      │  │统计分析     │  │                    │
└─────────────┘  └────────────┘  └────────────────────┘
     │                 │                  │
     └─────────────────┴──────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                    共享数据层                            │
│  PostgreSQL + Milvus + Neo4j + Redis + RabbitMQ        │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 服务划分详解

### 2.1 Java微服务（Spring Boot 3.2）

#### 服务1: user-service (用户服务)

**职责**：
- 用户注册、登录
- JWT Token生成与验证
- 用户权限管理（RBAC）
- OAuth2第三方登录

**技术栈**：
```xml
<dependencies>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-security</artifactId>
    </dependency>
    <dependency>
        <groupId>io.jsonwebtoken</groupId>
        <artifactId>jjwt</artifactId>
    </dependency>
    <dependency>
        <groupId>org.mybatis.spring.boot</groupId>
        <artifactId>mybatis-spring-boot-starter</artifactId>
    </dependency>
</dependencies>
```

**核心接口**：
```java
@RestController
@RequestMapping("/api/v1/auth")
public class AuthController {

    @Autowired
    private UserService userService;

    @Autowired
    private JwtTokenProvider tokenProvider;

    @PostMapping("/register")
    public ResponseEntity<RegisterResponse> register(
        @RequestBody @Valid RegisterRequest request
    ) {
        User user = userService.register(request);
        return ResponseEntity.ok(new RegisterResponse(user));
    }

    @PostMapping("/login")
    public ResponseEntity<LoginResponse> login(
        @RequestBody @Valid LoginRequest request
    ) {
        User user = userService.authenticate(
            request.getEmail(),
            request.getPassword()
        );
        String token = tokenProvider.generateToken(user);
        return ResponseEntity.ok(
            LoginResponse.builder()
                .accessToken(token)
                .tokenType("Bearer")
                .expiresIn(3600)
                .user(UserDTO.from(user))
                .build()
        );
    }
}
```

---

#### 服务2: collection-service (知识库服务)

**职责**：
- 知识库CRUD
- 知识库权限分配
- 成员管理

**核心代码**：
```java
@RestController
@RequestMapping("/api/v1/collections")
public class CollectionController {

    @Autowired
    private CollectionService collectionService;

    @PostMapping
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<CollectionDTO> createCollection(
        @RequestBody @Valid CreateCollectionRequest request,
        @AuthenticationPrincipal UserDetails userDetails
    ) {
        Collection collection = collectionService.create(
            request,
            userDetails.getUserId()
        );
        return ResponseEntity.status(HttpStatus.CREATED)
            .body(CollectionDTO.from(collection));
    }

    @GetMapping
    public ResponseEntity<PageResponse<CollectionDTO>> listCollections(
        @RequestParam(defaultValue = "1") int page,
        @RequestParam(defaultValue = "20") int pageSize,
        @AuthenticationPrincipal UserDetails userDetails
    ) {
        Page<Collection> collections = collectionService.listByUser(
            userDetails.getUserId(),
            page,
            pageSize
        );
        return ResponseEntity.ok(
            PageResponse.from(collections, CollectionDTO::from)
        );
    }
}
```

---

#### 服务3: document-service (文档管理服务)

**职责**：
- 文档元数据管理
- 文档列表查询
- 文档删除（触发Python服务清理向量）

**核心代码**：
```java
@RestController
@RequestMapping("/api/v1/documents")
public class DocumentController {

    @Autowired
    private DocumentService documentService;

    @Autowired
    private IngestionServiceClient ingestionServiceClient; // Feign客户端

    @PostMapping("/upload")
    @PreAuthorize("hasPermission(#request.collectionId, 'WRITE')")
    public ResponseEntity<DocumentDTO> uploadDocument(
        @RequestParam("file") MultipartFile file,
        @RequestParam("collectionId") Long collectionId,
        @RequestParam(value = "metadata", required = false) String metadataJson
    ) {
        // 1. 保存文档元数据到PostgreSQL
        Document document = documentService.createDocument(
            file, collectionId, metadataJson
        );

        // 2. 异步调用Python服务处理文档
        ingestionServiceClient.processDocument(
            ProcessDocumentRequest.builder()
                .documentId(document.getId())
                .filePath(document.getStoragePath())
                .fileType(document.getFileType())
                .collectionId(collectionId)
                .build()
        );

        return ResponseEntity.status(HttpStatus.CREATED)
            .body(DocumentDTO.from(document));
    }

    @DeleteMapping("/{documentId}")
    public ResponseEntity<Void> deleteDocument(
        @PathVariable Long documentId
    ) {
        // 1. 删除PostgreSQL中的元数据
        documentService.delete(documentId);

        // 2. 异步调用Python服务删除向量
        ingestionServiceClient.deleteDocumentVectors(documentId);

        return ResponseEntity.noContent().build();
    }
}
```

---

#### 服务4: gateway-service (API网关)

**职责**：
- 统一路由
- JWT认证
- 限流熔断
- 日志记录

**配置**：
```yaml
# application.yml
spring:
  cloud:
    gateway:
      routes:
        # Java服务路由
        - id: user-service
          uri: lb://user-service
          predicates:
            - Path=/api/v1/auth/**, /api/v1/users/**
          filters:
            - name: RequestRateLimiter
              args:
                redis-rate-limiter.replenishRate: 10
                redis-rate-limiter.burstCapacity: 20

        - id: collection-service
          uri: lb://collection-service
          predicates:
            - Path=/api/v1/collections/**

        # Python服务路由
        - id: chat-service
          uri: lb://rag-service
          predicates:
            - Path=/api/v1/chat/**, /api/v1/search/**
          filters:
            - name: Retry
              args:
                retries: 3
                statuses: BAD_GATEWAY

        - id: graph-service
          uri: lb://graph-service
          predicates:
            - Path=/api/v1/graph/**

      # 全局过滤器
      default-filters:
        - DedupeResponseHeader=Access-Control-Allow-Origin
        - name: GlobalJwtAuthFilter
```

**JWT认证过滤器**：
```java
@Component
public class GlobalJwtAuthFilter implements GlobalFilter, Ordered {

    @Autowired
    private JwtTokenProvider tokenProvider;

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        String token = extractToken(exchange.getRequest());

        if (token != null && tokenProvider.validateToken(token)) {
            Long userId = tokenProvider.getUserIdFromToken(token);

            // 将用户信息添加到请求头，传递给下游服务
            ServerHttpRequest modifiedRequest = exchange.getRequest()
                .mutate()
                .header("X-User-Id", String.valueOf(userId))
                .build();

            return chain.filter(
                exchange.mutate().request(modifiedRequest).build()
            );
        }

        return chain.filter(exchange);
    }

    @Override
    public int getOrder() {
        return -100; // 高优先级
    }
}
```

---

### 2.2 Python微服务（FastAPI）

#### 服务1: ingestion-service (文档摄入服务)

**职责**：
- 文档解析（PDF、Markdown、图片OCR）
- 文档分块（Chunking）
- Embedding生成
- 向量存储到Milvus

**核心代码**：
```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

app = FastAPI(title="Document Ingestion Service")

class ProcessDocumentRequest(BaseModel):
    document_id: int
    file_path: str
    file_type: str
    collection_id: int

@app.post("/api/v1/ingestion/process")
async def process_document(
    request: ProcessDocumentRequest,
    background_tasks: BackgroundTasks
):
    """接收Java服务的文档处理请求"""

    # 添加到后台任务队列
    background_tasks.add_task(
        process_document_task,
        request.document_id,
        request.file_path,
        request.file_type,
        request.collection_id
    )

    return {
        "document_id": request.document_id,
        "status": "processing"
    }

async def process_document_task(
    document_id: int,
    file_path: str,
    file_type: str,
    collection_id: int
):
    """后台处理文档"""

    # 1. 解析文档
    parser = ParserFactory.create(file_type)
    content = parser.parse(file_path)

    # 2. 分块
    chunker = SemanticChunker()
    chunks = chunker.split(content.text, chunk_size=512)

    # 3. 生成Embedding
    embeddings = await embedding_service.batch_embed(
        [chunk.text for chunk in chunks]
    )

    # 4. 存储到Milvus
    await milvus_client.insert(
        collection_name=f"collection_{collection_id}",
        data=[
            {
                "chunk_id": chunk.id,
                "document_id": document_id,
                "embedding": emb,
                "text": chunk.text
            }
            for chunk, emb in zip(chunks, embeddings)
        ]
    )

    # 5. 通知Java服务处理完成
    await notify_java_service(document_id, "completed")
```

---

#### 服务2: rag-service (RAG服务)

**职责**：
- LangGraph Agent编排
- 混合检索
- LLM生成答案

**核心代码**：
```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langgraph.graph import StateGraph

app = FastAPI(title="RAG Service")

@app.post("/api/v1/chat")
async def chat(request: ChatRequest):
    """同步问答"""
    result = await agent.ainvoke({
        "query": request.query,
        "collection_id": request.collection_id
    })

    return {
        "answer": result["answer"],
        "sources": result["retrieved_docs"],
        "metadata": {
            "confidence": result["confidence"],
            "iterations": result["iteration"]
        }
    }

@app.post("/api/v1/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式问答"""

    async def generate():
        async for chunk in agent.astream({
            "query": request.query,
            "collection_id": request.collection_id
        }):
            yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

---

## 3. 服务间通信

### 3.1 Java → Python（REST调用）

使用 **Spring Cloud OpenFeign** 调用Python服务：

```java
@FeignClient(
    name = "ingestion-service",
    url = "${services.ingestion.url}",
    fallback = IngestionServiceFallback.class
)
public interface IngestionServiceClient {

    @PostMapping("/api/v1/ingestion/process")
    void processDocument(@RequestBody ProcessDocumentRequest request);

    @DeleteMapping("/api/v1/ingestion/documents/{documentId}")
    void deleteDocumentVectors(@PathVariable Long documentId);
}

// 熔断降级
@Component
public class IngestionServiceFallback implements IngestionServiceClient {

    @Override
    public void processDocument(ProcessDocumentRequest request) {
        log.error("Ingestion service unavailable, document {} queued",
            request.getDocumentId());
        // 将任务放入消息队列延迟处理
        messageQueue.send("document.process", request);
    }
}
```

### 3.2 Python → Java（REST调用）

使用 **httpx** 异步调用Java服务：

```python
import httpx

class JavaServiceClient:
    def __init__(self, base_url: str):
        self.client = httpx.AsyncClient(base_url=base_url)

    async def notify_document_processed(
        self,
        document_id: int,
        status: str
    ):
        """通知Java服务文档处理完成"""
        response = await self.client.patch(
            f"/api/v1/documents/{document_id}/status",
            json={"status": status}
        )
        return response.json()

    async def get_user_info(self, user_id: int):
        """获取用户信息"""
        response = await self.client.get(
            f"/api/v1/users/{user_id}"
        )
        return response.json()
```

### 3.3 异步通信（消息队列）

使用 **RabbitMQ** 解耦服务：

**Java发送消息**：
```java
@Service
public class DocumentEventPublisher {

    @Autowired
    private RabbitTemplate rabbitTemplate;

    public void publishDocumentUploaded(Long documentId) {
        DocumentUploadedEvent event = DocumentUploadedEvent.builder()
            .documentId(documentId)
            .timestamp(Instant.now())
            .build();

        rabbitTemplate.convertAndSend(
            "document.exchange",
            "document.uploaded",
            event
        );
    }
}
```

**Python消费消息**：
```python
import aio_pika

async def consume_document_events():
    connection = await aio_pika.connect_robust("amqp://guest:guest@rabbitmq/")
    channel = await connection.channel()
    queue = await channel.declare_queue("document.uploaded")

    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            async with message.process():
                event = json.loads(message.body)
                await handle_document_uploaded(event["documentId"])
```

---

## 4. 共享数据层

### 4.1 数据库访问策略

**原则**：每个服务拥有自己的数据表，避免跨服务直接访问数据库

| 服务 | 拥有的表 | 访问方式 |
|------|---------|---------|
| user-service | users, roles, permissions | 直接访问 |
| collection-service | collections, collection_permissions | 直接访问 |
| document-service | documents, chunks (元数据) | 直接访问 |
| ingestion-service | - | 读取documents表（只读） |
| rag-service | conversations, messages | 直接访问 |

### 4.2 Milvus分区策略

按 `collection_id` 分区，避免跨知识库查询：

```python
# Python服务
collection = Collection("intelliknowledge_chunks")

# 创建分区
collection.create_partition(f"collection_{collection_id}")

# 插入时指定分区
collection.insert(
    data=[...],
    partition_name=f"collection_{collection_id}"
)

# 查询时只搜索对应分区
results = collection.search(
    data=[query_embedding],
    partition_names=[f"collection_{collection_id}"],
    limit=10
)
```

---

## 5. 部署架构

### 5.1 Docker Compose（开发环境）

```yaml
version: '3.8'

services:
  # ===== 基础设施 =====
  postgres:
    image: postgres:15
    ports: ["5432:5432"]
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  rabbitmq:
    image: rabbitmq:3-management
    ports: ["5672:5672", "15672:15672"]

  milvus:
    image: milvusdb/milvus:v2.3.0
    ports: ["19530:19530"]

  # ===== 注册中心 =====
  nacos:
    image: nacos/nacos-server:v2.2.3
    ports: ["8848:8848"]

  # ===== Java微服务 =====
  gateway-service:
    build: ./java-services/gateway-service
    ports: ["8080:8080"]
    environment:
      - SPRING_PROFILES_ACTIVE=dev
      - NACOS_SERVER=nacos:8848

  user-service:
    build: ./java-services/user-service
    environment:
      - SPRING_PROFILES_ACTIVE=dev
      - NACOS_SERVER=nacos:8848

  collection-service:
    build: ./java-services/collection-service
    environment:
      - SPRING_PROFILES_ACTIVE=dev
      - NACOS_SERVER=nacos:8848

  document-service:
    build: ./java-services/document-service
    environment:
      - SPRING_PROFILES_ACTIVE=dev
      - NACOS_SERVER=nacos:8848

  # ===== Python微服务 =====
  ingestion-service:
    build: ./python_services/ingestion-service
    ports: ["8001:8000"]
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/intelliknowledge
      - MILVUS_HOST=milvus

  rag-service:
    build: ./python_services/rag-service
    ports: ["8002:8000"]
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - MILVUS_HOST=milvus

  graph-service:
    build: ./python_services/graph-service
    ports: ["8003:8000"]
    environment:
      - NEO4J_URI=bolt://neo4j:7687

  # ===== 前端 =====
  frontend:
    build: ./frontend
    ports: ["3000:3000"]
```

### 5.2 Kubernetes（生产环境）

```yaml
# k8s/java-services-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gateway-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gateway-service
  template:
    metadata:
      labels:
        app: gateway-service
    spec:
      containers:
      - name: gateway
        image: intelliknowledge/gateway-service:latest
        ports:
        - containerPort: 8080
        env:
        - name: SPRING_PROFILES_ACTIVE
          value: "prod"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"

---
# k8s/python_services-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-service
spec:
  replicas: 5  # Python服务多实例
  selector:
    matchLabels:
      app: rag-service
  template:
    metadata:
      labels:
        app: rag-service
    spec:
      containers:
      - name: rag
        image: intelliknowledge/rag-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
```

---

## 6. 开发工作流

### 6.1 项目目录结构

```
IntelliKnowledge-RAG/
├── java-services/              # Java微服务
│   ├── gateway-service/
│   │   ├── src/
│   │   ├── pom.xml
│   │   └── Dockerfile
│   ├── user-service/
│   ├── collection-service/
│   └── document-service/
│
├── python-services/            # Python微服务
│   ├── ingestion-service/
│   │   ├── app/
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── rag-service/
│   └── graph-service/
│
├── frontend/                   # 前端
├── deployment/                 # 部署配置
│   ├── docker-compose.yml
│   └── kubernetes/
└── docs/                       # 文档
```

### 6.2 本地开发启动

```bash
# 1. 启动基础设施
docker-compose up -d postgres redis rabbitmq milvus nacos

# 2. 启动Java服务
cd java-services/gateway-service
mvn spring-boot:run

cd ../user-service
mvn spring-boot:run

# 3. 启动Python服务
cd python_services/ingestion-service
uvicorn app.main:app --reload --port 8001

cd ../rag-service
uvicorn app.main:app --reload --port 8002

# 4. 启动前端
cd frontend
npm run dev
```

---

## 7. 优势总结

### 7.1 技术优势

✅ **简历亮点**
- 微服务架构：Spring Cloud + FastAPI
- 跨语言协作：Java + Python
- 企业级技术栈：Spring Boot 3 + MyBatis-Plus + Nacos
- AI技术栈：LangGraph + LangChain + Milvus

✅ **性能优势**
- Java处理高并发业务请求（Gateway、用户认证）
- Python处理AI计算密集型任务
- 各司其职，性能最优

✅ **可维护性**
- 服务边界清晰
- 独立开发、测试、部署
- 技术栈升级互不影响

✅ **扩展性**
- 独立扩展：AI服务压力大时只扩展Python服务
- 新增功能灵活：AI相关用Python，业务相关用Java

### 7.2 学习价值

完成这个项目后，你将掌握：

1. **Java技术栈**
   - Spring Boot 3 新特性
   - Spring Cloud微服务全家桶
   - MyBatis-Plus ORM
   - Spring Security + JWT

2. **Python技术栈**
   - FastAPI异步编程
   - LangGraph Agent编排
   - Milvus向量数据库

3. **架构能力**
   - 微服务架构设计
   - 服务拆分原则
   - 跨语言服务通信
   - 分布式系统设计

4. **DevOps能力**
   - Docker容器化
   - Kubernetes编排
   - 服务监控与治理

---

## 8. 与纯Python方案对比

| 维度 | 纯Python方案 | Java+Python混合方案 |
|------|-------------|-------------------|
| **开发效率** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **性能** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **简历价值** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **企业认可度** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **技术深度** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **部署复杂度** | ⭐⭐⭐⭐ | ⭐⭐⭐ |

**建议**：
- 如果时间充裕，推荐 **Java+Python混合方案** ⭐⭐⭐⭐⭐
- 如果快速实现MVP，可以先用纯Python，后期重构为混合架构

---

## 9. 实施建议

### 阶段1：MVP（第1-4周）
**使用纯Python**，快速验证核心功能

### 阶段2：架构升级（第5-8周）
**引入Java微服务**，重构业务层
- Week 5: 搭建Spring Cloud基础框架
- Week 6: 实现user-service和gateway-service
- Week 7: 实现collection-service和document-service
- Week 8: 服务联调和测试

### 阶段3：持续优化（第9-16周）
- 完善监控
- 性能优化
- 补充文档

---

**文档版本**：v1.0
**最后更新**：2025-12-04
**作者**：Jerry
