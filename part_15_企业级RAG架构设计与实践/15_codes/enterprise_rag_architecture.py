"""
企业级RAG架构设计与实践
=============================

本模块演示企业级RAG系统的核心架构设计，包括：
- 高可用架构设计
- 多租户隔离
- 水平扩展策略
- 安全权限与审计
- 生产级监控

依赖安装：
    pip install llama-index llama-index-postprocessor-rerank tenacity cachetools langchain-community

Author: RAG Learning Course
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import time
import logging
from functools import wraps
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# 第一部分：核心数据结构和配置
# ============================================================

class TenantType(Enum):
    """
    租户类型枚举

    企业级RAG系统需要支持多租户隔离，不同租户的数据完全隔离。
    租户类型决定了资源分配策略和隔离级别。
    """
    FREE = "free"           # 免费租户，资源共享
    STANDARD = "standard"   # 标准租户，独立资源
    ENTERPRISE = "enterprise"  # 企业租户，完全隔离


@dataclass
class TenantContext:
    """
    租户上下文

    用于在多租户环境中跟踪请求的租户信息。
    每个租户有独立的配置、配额和数据命名空间。
    """
    tenant_id: str                    # 租户唯一标识
    tenant_type: TenantType = TenantType.STANDARD  # 租户类型
    quota_max_requests: int = 1000     # 每分钟最大请求数
    quota_max_documents: int = 10000   # 最大文档数
    quota_max_storage_gb: int = 10     # 最大存储空间(GB)
    rate_limit_rpm: int = 60          # 每分钟请求限制
    created_at: datetime = field(default_factory=datetime.now)

    def is_resource_exceeded(self, current_docs: int, current_storage_gb: int) -> bool:
        """检查是否超出资源配额"""
        return (current_docs >= self.quota_max_documents or
                current_storage_gb >= self.quota_max_storage_gb)


@dataclass
class AuditLogEntry:
    """
    审计日志条目

    生产环境需要完整的审计跟踪，记录所有关键操作。
    包括用户操作、数据访问、系统配置变更等。
    """
    timestamp: datetime
    tenant_id: str
    user_id: str
    action: str                    # 操作类型：QUERY, INDEX, DELETE, CONFIG_CHANGE
    resource_type: str             # 资源类型：DOCUMENT, CHUNK, INDEX
    resource_id: str               # 资源唯一标识
    query: Optional[str] = None    # 查询内容（如果是查询操作）
    success: bool = True
    error_message: Optional[str] = None
    latency_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadBalancerConfig:
    """
    负载均衡配置

    企业级系统需要多实例部署，负载均衡确保请求分发到健康实例。
    支持多种负载均衡策略和健康检查。
    """
    strategy: str = "round_robin"   # 负载策略：round_robin, least_connection, weighted
    health_check_interval: int = 30  # 健康检查间隔（秒）
    timeout_threshold: int = 5000   # 超时阈值（毫秒）
    max_retries: int = 3            # 最大重试次数


# ============================================================
# 第二部分：多租户隔离管理器
# ============================================================

class TenantIsolationManager:
    """
    多租户隔离管理器

    企业级RAG系统的核心组件，负责：
    1. 租户资源的完全隔离（数据、索引、配置）
    2. 租户配额的实施和限制
    3. 租户级别的访问控制

    设计要点：
    - 每个租户有独立的向量存储命名空间
    - 租户数据物理隔离而非逻辑隔离
    - 配额限制在API层和存储层双重实施
    """

    def __init__(self):
        # 租户注册表：存储所有租户的上下文信息
        self._tenants: Dict[str, TenantContext] = {}

        # 租户向量存储映射：一个租户可能使用多个向量存储（分片）
        self._vector_stores: Dict[str, List[str]] = {}

        # 租户配额使用情况追踪
        self._quota_usage: Dict[str, Dict[str, Any]] = {}

        # 租户级别的访问策略
        self._access_policies: Dict[str, Dict[str, bool]] = {}

        self._lock = threading.Lock()

    def register_tenant(self, tenant_id: str, tenant_type: TenantType = TenantType.STANDARD) -> TenantContext:
        """
        注册新租户

        Args:
            tenant_id: 租户唯一标识
            tenant_type: 租户类型，决定资源配额

        Returns:
            TenantContext: 租户上下文对象
        """
        with self._lock:
            # 根据租户类型设置默认配额
            quota_config = {
                TenantType.FREE: {"max_requests": 100, "max_documents": 100, "max_storage": 1},
                TenantType.STANDARD: {"max_requests": 1000, "max_documents": 10000, "max_storage": 10},
                TenantType.ENTERPRISE: {"max_requests": 10000, "max_documents": 1000000, "max_storage": 1000},
            }

            config = quota_config.get(tenant_type, quota_config[TenantType.STANDARD])

            tenant_ctx = TenantContext(
                tenant_id=tenant_id,
                tenant_type=tenant_type,
                quota_max_requests=config["max_requests"],
                quota_max_documents=config["max_documents"],
                quota_max_storage_gb=config["max_storage"],
            )

            self._tenants[tenant_id] = tenant_ctx
            self._vector_stores[tenant_id] = []
            self._quota_usage[tenant_id] = {
                "documents": 0,
                "storage_gb": 0,
                "requests_today": 0,
                "last_request_time": None,
            }

            logging.info(f"新租户注册成功: {tenant_id}, 类型: {tenant_type.value}")
            return tenant_ctx

    def get_tenant_context(self, tenant_id: str) -> Optional[TenantContext]:
        """获取租户上下文"""
        return self._tenants.get(tenant_id)

    def check_quota(self, tenant_id: str, operation: str = "request") -> bool:
        """
        检查租户配额是否允许操作

        Args:
            tenant_id: 租户ID
            operation: 操作类型

        Returns:
            bool: True表示配额允许，False表示超出配额
        """
        if tenant_id not in self._tenants:
            return False

        tenant = self._tenants[tenant_id]
        usage = self._quota_usage.get(tenant_id, {})

        if operation == "request":
            # 检查请求配额（简化实现，实际需要滑动窗口）
            requests_today = usage.get("requests_today", 0)
            return requests_today < tenant.quota_max_requests

        elif operation == "document":
            doc_count = usage.get("documents", 0)
            return doc_count < tenant.quota_max_documents

        elif operation == "storage":
            storage = usage.get("storage_gb", 0)
            return storage < tenant.quota_max_storage_gb

        return True

    def record_usage(self, tenant_id: str, operation: str, value: Any = 1):
        """记录租户使用量"""
        with self._lock:
            if tenant_id in self._quota_usage:
                if operation in self._quota_usage[tenant_id]:
                    self._quota_usage[tenant_id][operation] += value

    def get_tenant_vector_namespace(self, tenant_id: str) -> str:
        """
        获取租户独立的向量存储命名空间

        确保不同租户的数据完全隔离，命名空间隔离是
        最基础的多租户隔离策略。
        """
        return f"tenant_{tenant_id}_vector_store"

    def delete_tenant(self, tenant_id: str) -> bool:
        """
        删除租户及其所有关联数据

        这是一个危险操作，需要：
        1. 确认调用者有权限
        2. 先删除所有向量存储数据
        3. 清理配额使用记录
        4. 发送审计日志
        """
        with self._lock:
            if tenant_id not in self._tenants:
                return False

            # 清理向量存储（实际实现需要调用存储后端删除）
            vector_namespaces = self._vector_stores.get(tenant_id, [])
            for ns in vector_namespaces:
                # 在实际实现中，这里会调用向量存储的删除接口
                logging.info(f"清理租户向量存储: {ns}")

            # 从注册表移除
            del self._tenants[tenant_id]
            del self._vector_stores[tenant_id]
            del self._quota_usage[tenant_id]
            del self._access_policies[tenant_id]

            logging.info(f"租户删除成功: {tenant_id}")
            return True


# ============================================================
# 第三部分：审计日志服务
# ============================================================

class AuditLogger:
    """
    审计日志服务

    企业级系统的核心组件，负责记录所有敏感操作：
    1. 数据访问（谁在什么时间访问了什么数据）
    2. 数据修改（文档的增删改操作）
    3. 系统配置变更
    4. 安全相关事件（认证失败、权限检查失败）

    设计要点：
    - 审计日志需要持久化存储（生产环境应写入专门的审计日志系统）
    - 日志内容需包含足够的上下文用于事后分析
    - 高并发环境下需要异步写入和批量处理
    """

    def __init__(self, retention_days: int = 90):
        """
        初始化审计日志服务

        Args:
            retention_days: 日志保留天数（超过后可以归档或删除）
        """
        self._logs: List[AuditLogEntry] = []
        self._retention_days = retention_days
        self._write_lock = threading.Lock()

        # 异步写入队列（生产环境应使用专门的日志队列）
        self._async_queue: List[AuditLogEntry] = []
        self._flush_interval = 5  # 秒

    def log(self, entry: AuditLogEntry):
        """
        记录审计日志

        Args:
            entry: 审计日志条目
        """
        # 生产环境应该异步写入，这里简化为同步
        with self._write_lock:
            # 添加系统级别的字段
            entry.metadata["server_name"] = "rag-server-01"
            entry.metadata["environment"] = "production"
            entry.metadata["log_id"] = self._generate_log_id(entry)

            self._logs.append(entry)

            # 生产环境应该发送到专门的日志系统（ELK、Splunk等）
            logging.debug(f"审计日志记录: {entry.action} - {entry.tenant_id}/{entry.user_id}")

    def _generate_log_id(self, entry: AuditLogEntry) -> str:
        """生成唯一的日志ID"""
        content = f"{entry.timestamp}{entry.tenant_id}{entry.action}{entry.resource_id}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def query_logs(
        self,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditLogEntry]:
        """
        查询审计日志

        支持多种过滤条件组合，用于：
        - 合规审计：查找特定用户的数据访问记录
        - 故障排查：查找特定时间段的操作
        - 安全分析：查找异常访问模式

        Args:
            tenant_id: 按租户过滤
            user_id: 按用户过滤
            action: 按操作类型过滤
            start_time: 开始时间
            end_time: 结束时间
            limit: 返回结果数量限制

        Returns:
            符合条件的审计日志条目列表
        """
        results = []

        for entry in self._logs:
            # 多条件过滤
            if tenant_id and entry.tenant_id != tenant_id:
                continue
            if user_id and entry.user_id != user_id:
                continue
            if action and entry.action != action:
                continue
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue

            results.append(entry)

            if len(results) >= limit:
                break

        return results

    def get_security_report(self, tenant_id: str, days: int = 7) -> Dict[str, Any]:
        """
        生成安全报告

        汇总指定时间段内的安全相关事件，用于：
        - 定期安全审计
        - 异常检测
        - 合规报告
        """
        end_time = datetime.now()
        start_time = datetime.now() - timedelta(days=days)

        logs = self.query_logs(tenant_id=tenant_id, start_time=start_time, end_time=end_time, limit=10000)

        # 统计各类事件
        action_counts = {}
        failed_actions = []

        for entry in logs:
            action_counts[entry.action] = action_counts.get(entry.action, 0) + 1
            if not entry.success:
                failed_actions.append(entry)

        return {
            "tenant_id": tenant_id,
            "report_period": f"{start_time} to {end_time}",
            "total_events": len(logs),
            "action_breakdown": action_counts,
            "failed_actions": len(failed_actions),
            "failed_action_details": failed_actions[-10:],  # 最近10个失败事件
        }


# ============================================================
# 第四部分：高可用架构
# ============================================================

@dataclass
class HealthCheckResult:
    """健康检查结果"""
    component: str
    healthy: bool
    latency_ms: int
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


class HealthCheckManager:
    """
    健康检查管理器

    企业级系统需要持续监控各组件的健康状态：
    1. 向量存储可用性
    2. LLM服务可用性
    3. 嵌入模型服务可用性
    4. 网络延迟检查

    健康检查结果用于：
    - 负载均衡器判断实例是否健康
    - 自动故障转移
    - 告警系统触发
    """

    def __init__(self):
        self._components: Dict[str, Callable[[], bool]] = {}
        self._last_results: Dict[str, HealthCheckResult] = {}
        self._check_interval = 30  # 秒

    def register_component(self, name: str, check_func: Callable[[], bool]):
        """
        注册需要健康检查的组件

        Args:
            name: 组件名称
            check_func: 返回True表示健康，False表示不健康
        """
        self._components[name] = check_func

    def check_all(self) -> List[HealthCheckResult]:
        """
        执行所有组件的健康检查

        Returns:
            健康检查结果列表
        """
        results = []

        for name, check_func in self._components.items():
            start_time = time.time()
            try:
                is_healthy = check_func()
                latency_ms = int((time.time() - start_time) * 1000)

                result = HealthCheckResult(
                    component=name,
                    healthy=is_healthy,
                    latency_ms=latency_ms,
                    message="OK" if is_healthy else "Component check failed",
                )
            except Exception as e:
                result = HealthCheckResult(
                    component=name,
                    healthy=False,
                    latency_ms=0,
                    message=f"Health check error: {str(e)}",
                )

            self._last_results[name] = result
            results.append(result)

        return results

    def is_system_healthy(self) -> bool:
        """判断系统整体是否健康（所有组件都健康才算健康）"""
        if not self._last_results:
            return True
        return all(r.healthy for r in self._last_results.values())


class CircuitBreaker:
    """
    断路器模式实现

    用于防止级联故障：当某个服务连续失败超过阈值时，
    断路器打开，后续请求直接拒绝而不是继续调用失败的服务。

    状态转换：
    CLOSED（关闭）-> OPEN（打开）-> HALF_OPEN（半开）

    - CLOSED：正常状态，所有请求都通过
    - OPEN：服务不可用，快速拒绝请求
    - HALF_OPEN：尝试恢复，部分请求通过测试
    """

    class State(Enum):
        CLOSED = "closed"
        OPEN = "open"
        HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        """
        初始化断路器

        Args:
            failure_threshold: 连续失败多少次后打开断路器
            timeout_seconds: 断路器打开后多少秒尝试半开状态
        """
        self._failure_threshold = failure_threshold
        self._timeout_seconds = timeout_seconds
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._state = self.State.CLOSED
        self._lock = threading.Lock()

    @property
    def state(self) -> State:
        """获取当前断路器状态"""
        with self._lock:
            if self._state == self.State.OPEN:
                # 检查是否应该转换到半开状态
                if time.time() - self._last_failure_time >= self._timeout_seconds:
                    self._state = self.State.HALF_OPEN
            return self._state

    def allow_request(self) -> bool:
        """检查是否允许请求通过"""
        with self._lock:
            if self._state == self.State.CLOSED:
                return True
            elif self._state == self.State.HALF_OPEN:
                return True  # 半开状态允许部分请求通过测试
            else:  # OPEN
                return False

    def record_success(self):
        """记录成功调用"""
        with self._lock:
            self._failure_count = 0
            if self._state == self.State.HALF_OPEN:
                self._state = self.State.CLOSED
                logging.info("Circuit breaker recovered: CLOSED -> CLOSED")

    def record_failure(self):
        """记录失败调用"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._failure_count >= self._failure_threshold:
                self._state = self.State.OPEN
                logging.warning(f"Circuit breaker opened: failure_count={self._failure_count}")
            elif self._state == self.State.HALF_OPEN:
                self._state = self.State.OPEN
                logging.warning("Circuit breaker re-opened from HALF_OPEN")


# ============================================================
# 第五部分：企业级RAG系统主类
# ============================================================

class EnterpriseRAGSystem:
    """
    企业级RAG系统主类

    整合所有企业级特性，构建完整的生产级RAG系统：
    1. 多租户隔离
    2. 高可用架构
    3. 安全与审计
    4. 监控与告警
    5. 水平扩展支持

    架构设计原则：
    - 无单点故障：所有关键组件都有冗余
    - 优雅降级：部分组件故障时系统仍可提供有限服务
    - 可观测性：完整日志、指标、追踪
    - 成本控制：按需扩展，自动缩容
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化企业级RAG系统

        Args:
            config: 系统配置字典，包含:
                - vector_store_type: 向量存储类型
                - embedding_model: 嵌入模型配置
                - llm_config: LLM配置
                - load_balancer: 负载均衡配置
        """
        self.config = config

        # 核心组件初始化
        self._tenant_manager = TenantIsolationManager()
        self._audit_logger = AuditLogger()
        self._health_checker = HealthCheckManager()

        # 断路器：保护外部服务调用
        self._llm_circuit_breaker = CircuitBreaker(failure_threshold=5, timeout_seconds=60)
        self._vector_store_circuit_breaker = CircuitBreaker(failure_threshold=3, timeout_seconds=30)

        # 请求处理线程池
        self._executor = ThreadPoolExecutor(max_workers=config.get("max_workers", 10))

        # 初始化健康检查
        self._setup_health_checks()

        logging.info("企业级RAG系统初始化完成")

    def _setup_health_checks(self):
        """设置系统健康检查"""
        # 向量存储健康检查（示例）
        def check_vector_store() -> bool:
            # 实际实现应该执行实际的健康查询
            return True

        # LLM服务健康检查（示例）
        def check_llm_service() -> bool:
            # 实际实现应该发送测试请求
            return True

        self._health_checker.register_component("vector_store", check_vector_store)
        self._health_checker.register_component("llm_service", check_llm_service)
        self._health_checker.register_component("embedding_service", check_llm_service)

    def query(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        top_k: int = 5,
        use_reranker: bool = True,
    ) -> Dict[str, Any]:
        """
        处理用户查询

        这是RAG系统的核心入口，需要：
        1. 验证租户权限和配额
        2. 执行检索和生成流程
        3. 记录审计日志
        4. 处理异常和超时

        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            query: 用户查询
            top_k: 返回的相关文档数量
            use_reranker: 是否使用重排序

        Returns:
            包含答案和来源的字典
        """
        start_time = time.time()
        latency_ms = 0

        try:
            # 1. 配额检查
            if not self._tenant_manager.check_quota(tenant_id, "request"):
                return {
                    "success": False,
                    "error": "配额超限，请升级您的计划",
                    "error_code": "QUOTA_EXCEEDED",
                }

            # 2. 检查LLM断路器
            if not self._llm_circuit_breaker.allow_request():
                return {
                    "success": False,
                    "error": "服务暂时不可用，请稍后重试",
                    "error_code": "SERVICE_UNAVAILABLE",
                }

            # 3. 模拟检索过程（实际实现应调用真实的RAG组件）
            retrieved_context = self._mock_retrieve(query, top_k, tenant_id)
            latency_ms = int((time.time() - start_time) * 1000)

            # 4. 模拟LLM生成（实际实现应调用LLM API）
            answer = self._mock_generate(query, retrieved_context)

            # 5. 记录成功
            self._llm_circuit_breaker.record_success()
            self._tenant_manager.record_usage(tenant_id, "requests_today", 1)

            # 6. 记录审计日志
            self._audit_logger.log(AuditLogEntry(
                timestamp=datetime.now(),
                tenant_id=tenant_id,
                user_id=user_id,
                action="QUERY",
                resource_type="query",
                resource_id=self._generate_query_id(query),
                query=query,
                success=True,
                latency_ms=latency_ms,
            ))

            return {
                "success": True,
                "answer": answer,
                "sources": retrieved_context,
                "latency_ms": latency_ms,
            }

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            self._llm_circuit_breaker.record_failure()

            # 记录失败日志
            self._audit_logger.log(AuditLogEntry(
                timestamp=datetime.now(),
                tenant_id=tenant_id,
                user_id=user_id,
                action="QUERY",
                resource_type="query",
                resource_id="error",
                query=query,
                success=False,
                error_message=str(e),
                latency_ms=latency_ms,
            ))

            return {
                "success": False,
                "error": f"处理查询时发生错误: {str(e)}",
                "error_code": "INTERNAL_ERROR",
                "latency_ms": latency_ms,
            }

    def _mock_retrieve(self, query: str, top_k: int, tenant_id: str) -> List[Dict[str, Any]]:
        """
        模拟检索过程

        实际实现中，这里应该：
        1. 将查询向量化
        2. 在租户独立的向量存储中检索
        3. 可选：执行重排序
        """
        # 模拟返回一些检索结果
        return [
            {
                "content": f"这是关于'{query}'的相关文档内容片段...",
                "source": f"doc_{i+1}.pdf",
                "score": 0.95 - i * 0.1,
            }
            for i in range(min(top_k, 3))
        ]

    def _mock_generate(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        模拟LLM生成过程

        实际实现中，这里应该：
        1. 构建提示模板
        2. 调用LLM API
        3. 处理流式响应
        """
        sources_text = "\n".join([c["content"] for c in context])
        return f"根据检索到的资料，关于'{query}'的回答如下：\n{sources_text}\n\n（这是模拟的LLM响应）"

    def _generate_query_id(self, query: str) -> str:
        """生成查询ID"""
        return hashlib.sha256(f"{query}{time.time()}".encode()).hexdigest()[:16]

    def index_document(
        self,
        tenant_id: str,
        user_id: str,
        document_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        将文档索引到向量存储

        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            document_id: 文档唯一标识
            content: 文档内容
            metadata: 文档元数据

        Returns:
            索引操作结果
        """
        try:
            # 1. 配额检查
            if not self._tenant_manager.check_quota(tenant_id, "document"):
                return {
                    "success": False,
                    "error": "文档数量配额超限",
                    "error_code": "QUOTA_EXCEEDED",
                }

            # 2. 模拟文档向量化（实际实现应调用嵌入模型）
            chunks = self._split_into_chunks(content)

            # 3. 模拟存储到向量数据库
            # 实际实现应该使用租户独立的命名空间
            namespace = self._tenant_manager.get_tenant_vector_namespace(tenant_id)

            # 4. 记录审计日志
            self._audit_logger.log(AuditLogEntry(
                timestamp=datetime.now(),
                tenant_id=tenant_id,
                user_id=user_id,
                action="INDEX",
                resource_type="DOCUMENT",
                resource_id=document_id,
                metadata={"chunks_count": len(chunks), "namespace": namespace},
            ))

            # 5. 更新配额使用
            self._tenant_manager.record_usage(tenant_id, "documents", 1)
            self._tenant_manager.record_usage(tenant_id, "storage_gb", len(content) / (1024 * 1024 * 1024))

            return {
                "success": True,
                "document_id": document_id,
                "chunks_created": len(chunks),
                "namespace": namespace,
            }

        except Exception as e:
            self._audit_logger.log(AuditLogEntry(
                timestamp=datetime.now(),
                tenant_id=tenant_id,
                user_id=user_id,
                action="INDEX",
                resource_type="DOCUMENT",
                resource_id=document_id,
                success=False,
                error_message=str(e),
            ))

            return {
                "success": False,
                "error": f"索引文档时发生错误: {str(e)}",
                "error_code": "INDEX_ERROR",
            }

    def _split_into_chunks(self, content: str, chunk_size: int = 500) -> List[str]:
        """
        模拟文档分割

        实际实现应使用更智能的分割策略（如语义分割、结构感知分割）
        """
        words = content.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i:i + chunk_size]))
        return chunks if chunks else [content]

    def get_tenant_stats(self, tenant_id: str) -> Dict[str, Any]:
        """
        获取租户统计信息

        用于租户管理后台，显示资源使用情况
        """
        tenant = self._tenant_manager.get_tenant_context(tenant_id)
        usage = self._tenant_manager._quota_usage.get(tenant_id, {})

        if not tenant:
            return {"error": "租户不存在"}

        return {
            "tenant_id": tenant_id,
            "tenant_type": tenant.tenant_type.value,
            "usage": {
                "documents": usage.get("documents", 0),
                "storage_gb": usage.get("storage_gb", 0),
                "requests_today": usage.get("requests_today", 0),
            },
            "quotas": {
                "max_documents": tenant.quota_max_documents,
                "max_storage_gb": tenant.quota_max_storage_gb,
                "max_requests": tenant.quota_max_requests,
            },
            "utilization_rates": {
                "documents_pct": usage.get("documents", 0) / tenant.quota_max_documents * 100,
                "storage_pct": usage.get("storage_gb", 0) / tenant.quota_max_storage_gb * 100,
                "requests_pct": usage.get("requests_today", 0) / tenant.quota_max_requests * 100,
            },
        }

    def get_system_health(self) -> Dict[str, Any]:
        """
        获取系统健康状态

        用于运维监控和告警系统
        """
        health_results = self._health_checker.check_all()

        return {
            "healthy": self._health_checker.is_system_healthy(),
            "timestamp": datetime.now().isoformat(),
            "components": [
                {
                    "name": r.component,
                    "healthy": r.healthy,
                    "latency_ms": r.latency_ms,
                    "message": r.message,
                }
                for r in health_results
            ],
            "circuit_breakers": {
                "llm": self._llm_circuit_breaker.state.value,
                "vector_store": self._vector_store_circuit_breaker.state.value,
            },
        }

    def shutdown(self):
        """
        优雅关闭系统

        关闭线程池、刷新日志、清理资源
        """
        logging.info("开始优雅关闭企业级RAG系统...")
        self._executor.shutdown(wait=True)
        logging.info("系统关闭完成")


# ============================================================
# 第六部分：监控指标收集器
# ============================================================

class MetricsCollector:
    """
    监控指标收集器

    企业级系统需要持续监控以下指标：
    1. 请求延迟（P50、P95、P99）
    2. 请求量（QPS）
    3. 错误率
    4. 资源利用率（CPU、内存、GPU）
    5. 业务指标（检索召回率、答案质量）

    这些指标用于：
    - 实时告警
    - 容量规划
    - 性能优化决策
    """

    def __init__(self):
        self._metrics: Dict[str, List[float]] = {
            "query_latency_ms": [],
            "index_latency_ms": [],
            "request_errors": 0,
            "request_success": 0,
        }
        self._lock = threading.Lock()

    def record_query_latency(self, latency_ms: int):
        """记录查询延迟"""
        with self._lock:
            self._metrics["query_latency_ms"].append(float(latency_ms))
            # 保留最近10000条记录
            if len(self._metrics["query_latency_ms"]) > 10000:
                self._metrics["query_latency_ms"] = self._metrics["query_latency_ms"][-10000:]

    def record_success(self):
        """记录成功请求"""
        with self._lock:
            self._metrics["request_success"] += 1

    def record_error(self):
        """记录失败请求"""
        with self._lock:
            self._metrics["request_errors"] += 1

    def get_percentile(self, metric_name: str, percentile: int) -> float:
        """计算百分位数"""
        with self._lock:
            values = self._metrics.get(metric_name, [])
            if not values:
                return 0.0
            sorted_values = sorted(values)
            index = int(len(sorted_values) * percentile / 100)
            return sorted_values[min(index, len(sorted_values) - 1)]

    def get_error_rate(self) -> float:
        """计算错误率"""
        with self._lock:
            total = self._metrics["request_success"] + self._metrics["request_errors"]
            if total == 0:
                return 0.0
            return self._metrics["request_errors"] / total

    def get_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        with self._lock:
            return {
                "query_latency_p50": self.get_percentile("query_latency_ms", 50),
                "query_latency_p95": self.get_percentile("query_latency_ms", 95),
                "query_latency_p99": self.get_percentile("query_latency_ms", 99),
                "request_success": self._metrics["request_success"],
                "request_errors": self._metrics["request_errors"],
                "error_rate": self.get_error_rate(),
            }


# ============================================================
# 第七部分：使用示例
# ============================================================

def main():
    """
    企业级RAG系统使用示例

    演示如何：
    1. 初始化系统
    2. 注册租户
    3. 执行查询和索引
    4. 获取统计信息
    """
    # 配置（实际生产环境应从配置文件加载）
    config = {
        "vector_store_type": "milvus",
        "embedding_model": "bge-large-zh",
        "llm_config": {"provider": "openai", "model": "gpt-4"},
        "max_workers": 20,
    }

    # 1. 初始化企业级RAG系统
    system = EnterpriseRAGSystem(config)

    # 2. 注册租户
    tenant_id = "tenant_001"
    system._tenant_manager.register_tenant(
        tenant_id=tenant_id,
        tenant_type=TenantType.STANDARD,
    )

    print("=" * 60)
    print("企业级RAG架构设计与实践 - 代码演示")
    print("=" * 60)

    # 3. 索引文档示例
    print("\n[1] 文档索引演示")
    doc_result = system.index_document(
        tenant_id=tenant_id,
        user_id="user_001",
        document_id="doc_001",
        content="这是一个测试文档的内容，用于演示企业级RAG系统的文档索引功能。",
        metadata={"source": "manual", "category": "test"},
    )
    print(f"索引结果: {doc_result}")

    # 4. 查询演示
    print("\n[2] 查询演示")
    query_result = system.query(
        tenant_id=tenant_id,
        user_id="user_001",
        query="RAG系统架构设计",
        top_k=3,
    )
    print(f"查询结果: {query_result}")

    # 5. 租户统计
    print("\n[3] 租户统计信息")
    stats = system.get_tenant_stats(tenant_id)
    print(f"租户统计: {stats}")

    # 6. 系统健康检查
    print("\n[4] 系统健康状态")
    health = system.get_system_health()
    print(f"健康状态: {health}")

    # 7. 审计日志查询
    print("\n[5] 审计日志查询")
    logs = system._audit_logger.query_logs(tenant_id=tenant_id)
    print(f"审计日志条数: {len(logs)}")
    if logs:
        print(f"最新日志: action={logs[-1].action}, success={logs[-1].success}")

    # 8. 指标收集
    print("\n[6] 监控指标")
    metrics = MetricsCollector()
    metrics.record_query_latency(150)
    metrics.record_query_latency(200)
    metrics.record_success()
    summary = metrics.get_summary()
    print(f"指标摘要: {summary}")

    # 9. 多租户演示
    print("\n[7] 多租户隔离演示")
    tenant_2 = "tenant_002"
    system._tenant_manager.register_tenant(
        tenant_id=tenant_2,
        tenant_type=TenantType.FREE,  # 免费租户，配额更严格
    )

    # 不同租户的数据完全隔离
    result_1 = system.query(tenant_id, "user_001", "企业架构设计", top_k=3)
    result_2 = system.query(tenant_2, "user_002", "企业架构设计", top_k=3)
    print(f"租户1查询: {result_1['success']}")
    print(f"租户2查询: {result_2['success']}")

    # 10. 关闭系统
    system.shutdown()
    print("\n系统演示完成！")


if __name__ == "__main__":
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    main()