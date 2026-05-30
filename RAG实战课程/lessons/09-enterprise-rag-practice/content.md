# 第九课：企业级 RAG 实战

## 本节概述

本节课将学习如何将 RAG 从原型验证推进到企业级生产系统。你将掌握：

- RAG 架构演进路径
- 企业级架构设计要点
- 工程化最佳实践
- 安全与合规
- 监控与运维

**学习时长**：约 3-4 小时

---

## 9.1 RAG 架构演进

### 9.1.1 四阶段演进

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG 架构演进四阶段                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  阶段 1: 原型验证                                           │
│  ──────────────────                                        │
│  单机 + 文件向量库 + 简单脚本                               │
│  目标：快速验证可行性                                       │
│                                                             │
│  阶段 2: 开发测试                                           │
│  ──────────────────                                        │
│  API 化 + LangChain + Chroma                               │
│  目标：支持多用户、初步测试                                 │
│                                                             │
│  阶段 3: 生产部署                                           │
│  ──────────────────                                        │
│  微服务 + Milvus + 缓存 + 监控                             │
│  目标：高可用、可扩展                                       │
│                                                             │
│  阶段 4: 持续优化                                           │
│  ──────────────────                                        │
│  A/B 测试 + 用户反馈 + 自动化优化                          │
│  目标：效果持续提升                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.1.2 各阶段特征对比

| 维度 | 原型 | 开发测试 | 生产部署 | 持续优化 |
|------|------|----------|----------|----------|
| **部署** | 单机 | 单机/集群 | 多副本 | 多区域 |
| **向量库** | Chroma | Chroma | Milvus/Pinecone | 多向量库 |
| **LLM** | 共享 API | 共享 API | 专用实例 | 智能路由 |
| **监控** | 无 | 日志 | 完整监控 | 自动告警 |
| **容错** | 无 | 基础 | 自动恢复 | 自愈 |

---

## 9.2 企业级架构设计

### 9.2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户请求层                                │
│                   (Web / App / API Gateway)                     │
└─────────────────────────────┬───────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        API 网关层                                │
│     (认证 / 限流 / 路由 / 监控)                                  │
└─────────────────────────────┬───────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      RAG 服务层                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Query      │  │  Retrieval  │  │  Generation │            │
│  │  Processing │→ │  Service    │→ │  Service    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│         ↓                ↓                ↓                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Query      │  │  Vector     │  │  LLM        │            │
│  │  Cache      │  │  Index      │  │  Gateway    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────┬───────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        数据层                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Vector DB  │  │  Document   │  │  Cache      │            │
│  │  (Milvus)   │  │  Store      │  │  (Redis)   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2.2 核心组件设计

```python
# ========== 查询处理服务 ==========

class QueryProcessingService:
    """
    查询处理服务
    负责 Query 解析、扩展、路由
    """
    
    def __init__(self):
        self.query_rewriter = QueryRewriter()
        self.query_cache = RedisCache()
        self.intent_classifier = IntentClassifier()
    
    async def process(self, query: str, user_id: str) -> QueryContext:
        # 1. 检查缓存
        cached = await self.query_cache.get(f"{user_id}:{query}")
        if cached:
            return cached
        
        # 2. 意图分类
        intent = await self.intent_classifier.classify(query)
        
        # 3. Query 改写（针对不同意图）
        if intent == "factual":
            rewritten = await self.query_rewriter.rewrite_factual(query)
        elif intent == "analytical":
            rewritten = await self.query_rewriter.rewrite_analytical(query)
        else:
            rewritten = query
        
        # 4. 构建上下文
        context = QueryContext(
            original=query,
            rewritten=rewritten,
            intent=intent,
            user_id=user_id
        )
        
        # 5. 缓存
        await self.query_cache.set(
            f"{user_id}:{query}",
            context,
            ttl=3600
        )
        
        return context


# ========== 检索服务 ==========

class RetrievalService:
    """
    检索服务
    支持多向量库路由、混合检索
    """
    
    def __init__(self):
        self.vector_stores = {
            "product": MilvusStore(collection="product_docs"),
            "policy": MilvusStore(collection="policy_docs"),
            "faq": ChromaStore(collection="faq")
        }
        self.reranker = BGEReranker()
        self.hybrid_retriever = HybridRetriever()
    
    async def retrieve(
        self,
        query: QueryContext,
        top_k: int = 10
    ) -> List[RetrievalResult]:
        # 1. 确定检索目标（基于意图或显式指定）
        target_stores = self._get_target_stores(query)
        
        # 2. 并行初步召回
        recall_tasks = []
        for store_name in target_stores:
            recall_tasks.append(
                self.vector_stores[store_name].search(
                    query_vector=query.vector,
                    k=top_k * 2  # 多召回一些用于重排
                )
            )
        
        recall_results = await asyncio.gather(*recall_tasks)
        
        # 3. 扁平化 + 去重
        all_candidates = []
        seen = set()
        for results in recall_results:
            for r in results:
                key = r.doc.content[:50]
                if key not in seen:
                    seen.add(key)
                    all_candidates.append(r)
        
        # 4. Rerank
        reranked = await self.reranker.rerank(
            query=query.rewritten,
            candidates=all_candidates,
            top_k=top_k
        )
        
        return reranked
```

---

## 9.3 工程化最佳实践

### 9.3.1 缓存策略

```python
# ========== 多级缓存 ==========

class MultiLevelCache:
    """
    多级缓存
    L1: 内存缓存 (Dict)
    L2: Redis 缓存
    """
    
    def __init__(self):
        self.l1_cache = {}  # 内存缓存
        self.l2_cache = RedisCache()  # Redis 缓存
        self.l1_ttl = 60  # L1 缓存 60 秒
        self.l2_ttl = 3600  # L2 缓存 1 小时
    
    async def get(self, key: str) -> Optional[Any]:
        # L1 查找
        if key in self.l1_cache:
            return self.l1_cache[key]
        
        # L2 查找
        l2_value = await self.l2_cache.get(key)
        if l2_value:
            self.l1_cache[key] = l2_value
            return l2_value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None):
        self.l1_cache[key] = value
        
        if ttl is None:
            ttl = self.l2_ttl
        
        await self.l2_cache.set(key, value, ttl=ttl)


# ========== 缓存内容分类 ==========

CACHE_STRATEGY = {
    # 类型: (缓存内容, TTL)
    "query_result": ("完整问答结果", 3600),      # 1小时
    "retrieval": ("检索结果", 1800),            # 30分钟
    "embedding": ("向量", 86400),               # 1天
    "llm_response": ("LLM原始响应", 3600),     # 1小时
}
```

### 9.3.2 并发控制

```python
# ========== Semaphore 并发控制 ==========

class ConcurrencyController:
    """
    并发控制器
    防止向量库和 LLM 过载
    """
    
    def __init__(self):
        self.vector_semaphore = asyncio.Semaphore(100)  # 最多100个向量查询
        self.llm_semaphore = asyncio.Semaphore(50)    # 最多50个 LLM 调用
    
    async def vector_search(self, *args, **kwargs):
        async with self.vector_semaphore:
            return await vector_store.search(*args, **kwargs)
    
    async def llm_generate(self, *args, **kwargs):
        async with self.llm_semaphore:
            return await llm.chat(*args, **kwargs)


# ========== 熔断器 ==========

class CircuitBreaker:
    """
    熔断器
    防止级联故障
    """
    
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    async def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
            else:
                raise CircuitBreakerOpenError()
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise e
```

### 9.3.3 错误处理

```python
# ========== 分级错误处理 ==========

class RAGError(Exception):
    """RAG 系统基础异常"""
    pass

class RetrievalError(RAGError):
    """检索异常"""
    pass

class GenerationError(RAGError):
    """生成异常"""
    pass

class VectorStoreError(RAGError):
    """向量库异常"""
    pass

# ========== 降级策略 ==========

class GracefulDegradation:
    """
    优雅降级
    核心服务不可用时，提供备选方案
    """
    
    async def generate_with_fallback(
        self,
        query: str,
        context: List[Document]
    ) -> str:
        # 1. 尝试正常生成
        try:
            return await self.llm.chat(query, context)
        except RateLimitError:
            # 2. Rate Limit：尝试用缓存结果
            cached = await self.cache.get(f"llm:{hash(query)}")
            if cached:
                return cached
            # 3. 缓存也没有：使用简化模型
            return await self.fallback_model.generate(query, context)
        
        except TimeoutError:
            # 超时：使用缓存或简短回答
            cached = await self.cache.get(f"llm:{hash(query)}")
            if cached:
                return cached
            return "抱歉，当前服务繁忙，请稍后再试。"
```

---

## 9.4 安全与合规

### 9.4.1 数据安全

```python
# ========== 敏感信息过滤 ==========

class SensitiveDataFilter:
    """
    敏感数据过滤器
    在输入输出中过滤敏感信息
    """
    
    def __init__(self):
        self.patterns = [
            (r'\d{18}', '身份证号'),        # 身份证
            (r'\d{11}', '手机号'),          # 手机号
            (r'\d{4}-\d{4}-\d{4}-\d{4}', '银行卡号'),  # 银行卡
            (r'password[:\s]+.*', '密码'),
        ]
    
    def filter_input(self, text: str) -> str:
        """过滤输入中的敏感信息"""
        for pattern, name in self.patterns:
            text = re.sub(pattern, f'[{name}已脱敏]', text)
        return text
    
    def filter_output(self, text: str, metadata: Dict) -> str:
        """过滤输出中的敏感信息"""
        # 根据用户权限过滤
        if not metadata.get("can_see_salary"):
            text = re.sub(r'薪资[:\s]+\d+', '薪资 [已脱敏]', text)
        return text


# ========== 权限控制 ==========

class PermissionControl:
    """
    权限控制系统
    基于角色的访问控制 (RBAC)
    """
    
    def __init__(self):
        self.permissions = {
            "admin": ["*"],  # 管理员可访问所有
            "employee": ["public:*", "policy:*"],
            "customer": ["public:faq:*"]
        }
    
    def can_access(self, user_role: str, resource: str) -> bool:
        """检查用户是否有权限访问资源"""
        perms = self.permissions.get(user_role, [])
        
        if "*" in perms:
            return True
        
        for perm in perms:
            if perm.endswith(":*") and resource.startswith(perm[:-1]):
                return True
            if resource == perm:
                return True
        
        return False
```

### 9.4.2 审计日志

```python
# ========== 审计日志 ==========

class AuditLogger:
    """
    审计日志
    记录所有敏感操作
    """
    
    def __init__(self):
        self.logger = StructuredLogger("audit")
    
    async def log_query(
        self,
        user_id: str,
        query: str,
        result: str,
        metadata: Dict
    ):
        """记录问答日志"""
        await self.logger.info(
            "user_query",
            user_id=user_id,
            query_hash=hash(query),  # 不记录原始query
            result_length=len(result),
            retrieval_count=metadata.get("retrieval_count", 0),
            latency_ms=metadata.get("latency", 0),
            timestamp=datetime.now().isoformat()
        )
    
    async def log_admin_action(
        self,
        user_id: str,
        action: str,
        target: str,
        result: str
    ):
        """记录管理员操作"""
        await self.logger.warning(
            "admin_action",
            user_id=user_id,
            action=action,
            target=target,
            result=result,
            timestamp=datetime.now().isoformat()
        )
```

---

## 9.5 监控与运维

### 9.5.1 核心指标

```python
# ========== 监控系统集成 ==========

METRICS = {
    # 延迟指标
    "latency.query.p50": "Query 处理 P50 延迟",
    "latency.query.p99": "Query 处理 P99 延迟",
    "latency.retrieval.p50": "检索 P50 延迟",
    "latency.generation.p50": "生成 P50 延迟",
    
    # 吞吐量
    "qps": "每秒查询数",
    "concurrent_users": "并发用户数",
    
    # 质量指标
    "retrieval.empty_rate": "空召回率",
    "generation.error_rate": "生成错误率",
    "user.feedback.positive_rate": "用户好评率",
    
    # 系统指标
    "vectorstore.connection_pool.usage": "向量库连接池使用率",
    "llm.token_usage": "LLM Token 消耗",
    "cache.hit_rate": "缓存命中率",
}
```

### 9.5.2 健康检查

```python
# ========== 健康检查 ==========

class HealthChecker:
    """
    健康检查
    定期检查各组件状态
    """
    
    async def check(self) -> Dict:
        results = await asyncio.gather(
            self.check_vectorstore(),
            self.check_llm_gateway(),
            self.check_cache(),
            self.check_document_store(),
            return_exceptions=True
        )
        
        status = "healthy" if all(
            r == "ok" for r in results if not isinstance(r, Exception)
        ) else "degraded"
        
        return {
            "status": status,
            "components": {
                "vectorstore": results[0] if not isinstance(results[0], Exception) else "error",
                "llm": results[1] if not isinstance(results[1], Exception) else "error",
                "cache": results[2] if not isinstance(results[2], Exception) else "error",
                "document_store": results[3] if not isinstance(results[3], Exception) else "error",
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def check_vectorstore(self) -> str:
        try:
            await self.vectorstore.health_check()
            return "ok"
        except Exception as e:
            return f"error: {str(e)}"
```

---

## 本节总结

### 核心要点

1. **架构演进**：原型 → 开发 → 生产 → 持续优化
2. **工程化实践**：多级缓存、并发控制、优雅降级
3. **安全合规**：敏感过滤、权限控制、审计日志
4. **监控运维**：核心指标、健康检查、自动告警

### 代码文件

- `code/rag_system.py`: 企业级 RAG 系统框架
- `code/config.py`: 配置管理

### 思考题

1. 企业级 RAG 与原型系统的核心差异是什么？
2. 为什么需要多级缓存？Redis 缓存失效后会发生什么？
3. 熔断器的"半开"状态是什么设计意图？

### 下节预告

恭喜完成 RAG 实战课程！建议的下一步学习路径：
1. 动手实践：基于本课程代码构建自己的 RAG 系统
2. 深入特定领域：如 RAG + Agent、多模态 RAG
3. 性能优化：学习向量数据库调优、LLM 加速
