"""
配置管理
支持多环境配置、配置验证、敏感信息隐藏
"""

import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import json


class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class LLMConfig:
    """LLM 配置"""

    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 1000
    timeout: int = 60
    retry_count: int = 3

    # 速率限制
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000


@dataclass
class EmbeddingConfig:
    """Embedding 配置"""

    provider: str = "openai"
    model: str = "text-embedding-3-small"
    api_key: Optional[str] = None
    dimension: int = 1536
    batch_size: int = 100


@dataclass
class VectorStoreConfig:
    """向量库配置"""

    type: str = "chroma"  # chroma, milvus, pinecone, weaviate
    persist_directory: str = "./vector_db"
    collection_name: str = "documents"

    # 连接配置
    host: str = "localhost"
    port: int = 6333
    username: Optional[str] = None
    password: Optional[str] = None

    # 索引配置
    index_type: str = "hnsw"
    hnsw_space: str = "cosine"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200


@dataclass
class CacheConfig:
    """缓存配置"""

    enabled: bool = True
    type: str = "memory"  # memory, redis

    # Redis 配置
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0

    # TTL 配置（秒）
    query_result_ttl: int = 3600
    retrieval_ttl: int = 1800
    embedding_ttl: int = 86400


@dataclass
class SecurityConfig:
    """安全配置"""

    enable_sensitive_filter: bool = True
    enable_permission_check: bool = True
    enable_audit_log: bool = True

    # CORS
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    # 认证
    auth_type: str = "api_key"  # api_key, jwt, oauth


@dataclass
class MonitoringConfig:
    """监控配置"""

    enabled: bool = True
    log_level: str = "info"

    # 指标
    enable_metrics: bool = True
    metrics_port: int = 9090

    # 追踪
    enable_tracing: bool = False
    tracing_endpoint: Optional[str] = None


@dataclass
class Config:
    """完整配置"""

    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False

    # 组件配置
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # 应用配置
    app_name: str = "RAG System"
    app_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000

    # 并发控制
    max_concurrent_queries: int = 100
    max_concurrent_llm_calls: int = 50

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量加载配置"""
        env = Environment(os.getenv("RAG_ENV", "development"))
        debug = env == Environment.DEVELOPMENT

        return cls(
            environment=env,
            debug=debug,
            llm=LLMConfig(
                provider=os.getenv("LLM_PROVIDER", "openai"),
                model=os.getenv("LLM_MODEL", "gpt-4o"),
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("LLM_BASE_URL"),
            ),
            embedding=EmbeddingConfig(
                provider=os.getenv("EMBEDDING_PROVIDER", "openai"),
                model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
            vectorstore=VectorStoreConfig(
                type=os.getenv("VECTORSTORE_TYPE", "chroma"),
                persist_directory=os.getenv("VECTORSTORE_DIR", "./vector_db"),
            ),
            cache=CacheConfig(
                enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
                redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            ),
        )

    def to_dict(self, hide_sensitive: bool = True) -> Dict[str, Any]:
        """转换为字典（隐藏敏感信息）"""

        def mask(value: Any) -> Any:
            if isinstance(value, str) and len(value) > 8:
                return value[:4] + "****" + value[-4:]
            return value

        result = {}
        for key, value in self.__dict__.items():
            if key in ["api_key", "password", "secret"]:
                result[key] = "****" if hide_sensitive and value else None
            elif isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, dataclass):
                result[key] = value.__dict__
            else:
                result[key] = value

        return result

    def validate(self) -> List[str]:
        """验证配置，返回错误列表"""
        errors = []

        if not self.llm.api_key and self.llm.provider == "openai":
            errors.append("LLM API key is required when using OpenAI")

        if not self.embedding.api_key and self.embedding.provider == "openai":
            errors.append("Embedding API key is required when using OpenAI")

        if self.cache.enabled and self.cache.type == "redis":
            if not self.cache.redis_url:
                errors.append("Redis URL is required when using Redis cache")

        return errors


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 加载配置
    config = Config.from_env()

    # 验证配置
    errors = config.validate()
    if errors:
        print("配置错误:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("配置有效")
        print(json.dumps(config.to_dict(), indent=2, default=str))
