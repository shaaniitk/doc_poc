"""Production-ready configuration management system for LangGraph document processing."""

import os
import json
import yaml
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
from urllib.parse import urlparse


class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    MISTRAL = "mistral"


class ProcessingMode(Enum):
    """Processing execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    BATCH = "batch"
    STREAMING = "streaming"
    ADAPTIVE = "adaptive"


class ChunkingStrategy(Enum):
    """Document chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SENTENCE_AWARE = "sentence_aware"
    SEMANTIC = "semantic"
    PARAGRAPH = "paragraph"
    SLIDING_WINDOW = "sliding_window"
    ADAPTIVE = "adaptive"


class OutputFormat(Enum):
    """Supported output formats."""
    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"
    PDF = "pdf"
    HTML = "html"
    DOCX = "docx"


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 40000
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.api_key and self.provider != LLMProvider.LOCAL:
            # Try to get from environment
            env_key = f"{self.provider.value.upper()}_API_KEY"
            self.api_key = os.getenv(env_key)
            
            if not self.api_key:
                raise ValueError(f"API key required for {self.provider.value}")
    
    def get_client_config(self) -> Dict[str, Any]:
        """Get configuration for LLM client initialization."""
        config = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout
        }
        
        if self.api_key:
            config["api_key"] = self.api_key
        if self.api_base:
            config["api_base"] = self.api_base
        if self.api_version:
            config["api_version"] = self.api_version
            
        return config


@dataclass
class ChunkingConfig:
    """Document chunking configuration."""
    strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE_AWARE
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    preserve_sentences: bool = True
    preserve_paragraphs: bool = True
    use_semantic_splitting: bool = False
    semantic_threshold: float = 0.7
    
    def __post_init__(self):
        """Validate chunking configuration."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        
        if self.min_chunk_size > self.chunk_size:
            raise ValueError("Minimum chunk size cannot be greater than chunk size")
        
        if self.max_chunk_size < self.chunk_size:
            raise ValueError("Maximum chunk size cannot be less than chunk size")


@dataclass
class ProcessingConfig:
    """Document processing configuration."""
    mode: ProcessingMode = ProcessingMode.PARALLEL
    max_concurrent: int = 5
    batch_size: int = 10
    timeout_seconds: int = 300
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    enable_quality_checks: bool = True
    quality_threshold: float = 0.8
    
    def __post_init__(self):
        """Validate processing configuration."""
        if self.max_concurrent < 1:
            raise ValueError("Max concurrent must be at least 1")
        
        if self.batch_size < 1:
            raise ValueError("Batch size must be at least 1")
        
        if self.timeout_seconds < 1:
            raise ValueError("Timeout must be at least 1 second")


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration."""
    enable_monitoring: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None
    enable_metrics: bool = True
    metrics_interval: int = 60
    enable_health_checks: bool = True
    health_check_interval: int = 30
    enable_performance_tracking: bool = True
    memory_limit_mb: Optional[int] = None
    cpu_limit_percent: Optional[float] = None
    
    def __post_init__(self):
        """Validate monitoring configuration."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        
        self.log_level = self.log_level.upper()


@dataclass
class SecurityConfig:
    """Security configuration."""
    enable_input_validation: bool = True
    max_file_size_mb: int = 100
    allowed_file_types: List[str] = field(default_factory=lambda: [
        ".txt", ".pdf", ".docx", ".md", ".tex", ".rtf"
    ])
    sanitize_output: bool = True
    enable_rate_limiting: bool = True
    rate_limit_requests_per_minute: int = 60
    enable_audit_logging: bool = True
    mask_sensitive_data: bool = True
    
    def __post_init__(self):
        """Validate security configuration."""
        if self.max_file_size_mb < 1:
            raise ValueError("Max file size must be at least 1 MB")
        
        # Ensure file types start with dot
        self.allowed_file_types = [
            ext if ext.startswith('.') else f'.{ext}'
            for ext in self.allowed_file_types
        ]


@dataclass
class StorageConfig:
    """Storage configuration."""
    input_dir: Path = Path("input")
    output_dir: Path = Path("output")
    cache_dir: Path = Path("cache")
    temp_dir: Path = Path("temp")
    log_dir: Path = Path("logs")
    backup_dir: Optional[Path] = None
    enable_compression: bool = False
    cleanup_temp_files: bool = True
    max_cache_size_mb: int = 1000
    
    def __post_init__(self):
        """Create directories and validate storage configuration."""
        # Create directories if they don't exist
        for dir_path in [self.input_dir, self.output_dir, self.cache_dir, 
                        self.temp_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        if self.backup_dir:
            self.backup_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class DatabaseConfig:
    """Database configuration for knowledge graphs and caching."""
    enable_database: bool = False
    database_type: str = "sqlite"  # sqlite, postgresql, mongodb
    connection_string: Optional[str] = None
    host: str = "localhost"
    port: int = 5432
    database: str = "doc_processing"
    username: Optional[str] = None
    password: Optional[str] = None
    pool_size: int = 5
    max_overflow: int = 10
    
    def __post_init__(self):
        """Validate database configuration."""
        if self.enable_database and not self.connection_string:
            if self.database_type == "sqlite":
                self.connection_string = f"sqlite:///data/{self.database}.db"
            elif self.database_type == "postgresql":
                if not self.username or not self.password:
                    raise ValueError("Username and password required for PostgreSQL")
                self.connection_string = (
                    f"postgresql://{self.username}:{self.password}@"
                    f"{self.host}:{self.port}/{self.database}"
                )


@dataclass
class ApplicationConfig:
    """Main application configuration."""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    version: str = "1.0.0"
    
    # Component configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # Additional settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Set debug mode based on environment
        if self.environment == Environment.DEVELOPMENT:
            self.debug = True
        
        # Adjust configurations based on environment
        if self.environment == Environment.PRODUCTION:
            self.monitoring.log_level = "WARNING"
            self.security.enable_audit_logging = True
            self.processing.enable_caching = True
        elif self.environment == Environment.TESTING:
            self.monitoring.log_level = "DEBUG"
            self.processing.max_concurrent = 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def validate(self) -> List[str]:
        """Validate entire configuration and return list of issues."""
        issues = []
        
        # Validate LLM configuration
        if not self.llm.api_key and self.llm.provider != LLMProvider.LOCAL:
            issues.append(f"API key required for {self.llm.provider.value}")
        
        # Validate processing limits
        if self.processing.max_concurrent > 20:
            issues.append("Max concurrent processes should not exceed 20")
        
        # Validate security settings
        if self.environment == Environment.PRODUCTION:
            if not self.security.enable_input_validation:
                issues.append("Input validation should be enabled in production")
            if not self.security.enable_audit_logging:
                issues.append("Audit logging should be enabled in production")
        
        # Validate storage paths
        for path_name, path_value in [
            ("input_dir", self.storage.input_dir),
            ("output_dir", self.storage.output_dir),
            ("cache_dir", self.storage.cache_dir),
            ("temp_dir", self.storage.temp_dir),
            ("log_dir", self.storage.log_dir)
        ]:
            if not path_value.exists():
                issues.append(f"{path_name} does not exist: {path_value}")
        
        return issues


class ConfigManager:
    """Configuration manager for loading and managing application configuration."""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or Path("config.yaml")
        self.config: Optional[ApplicationConfig] = None
        self.logger = logging.getLogger("config_manager")
    
    def load_config(self, config_file: Optional[Path] = None) -> ApplicationConfig:
        """Load configuration from file or environment variables."""
        if config_file:
            self.config_file = config_file
        
        # Start with default configuration
        config_dict = {}
        
        # Load from file if it exists
        if self.config_file.exists():
            config_dict = self._load_from_file(self.config_file)
            self.logger.info(f"Loaded configuration from {self.config_file}")
        else:
            self.logger.warning(f"Configuration file {self.config_file} not found, using defaults")
        
        # Override with environment variables
        env_overrides = self._load_from_environment()
        config_dict = self._deep_merge(config_dict, env_overrides)
        
        # Create configuration object
        self.config = self._dict_to_config(config_dict)
        
        # Validate configuration
        issues = self.config.validate()
        if issues:
            self.logger.warning(f"Configuration validation issues: {issues}")
        
        return self.config
    
    def _load_from_file(self, file_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        with open(file_path, 'r') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f) or {}
            elif file_path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration overrides from environment variables."""
        env_config = {}
        
        # Environment
        if env_val := os.getenv("DOC_PROCESSING_ENV"):
            env_config["environment"] = env_val
        
        # LLM configuration
        llm_config = {}
        if env_val := os.getenv("LLM_PROVIDER"):
            llm_config["provider"] = env_val
        if env_val := os.getenv("LLM_MODEL"):
            llm_config["model"] = env_val
        if env_val := os.getenv("OPENAI_API_KEY"):
            llm_config["api_key"] = env_val
        if env_val := os.getenv("LLM_MAX_TOKENS"):
            llm_config["max_tokens"] = int(env_val)
        
        if llm_config:
            env_config["llm"] = llm_config
        
        # Processing configuration
        processing_config = {}
        if env_val := os.getenv("PROCESSING_MODE"):
            processing_config["mode"] = env_val
        if env_val := os.getenv("MAX_CONCURRENT"):
            processing_config["max_concurrent"] = int(env_val)
        
        if processing_config:
            env_config["processing"] = processing_config
        
        # Monitoring configuration
        monitoring_config = {}
        if env_val := os.getenv("LOG_LEVEL"):
            monitoring_config["log_level"] = env_val
        if env_val := os.getenv("ENABLE_MONITORING"):
            monitoring_config["enable_monitoring"] = env_val.lower() == "true"
        
        if monitoring_config:
            env_config["monitoring"] = monitoring_config
        
        return env_config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> ApplicationConfig:
        """Convert dictionary to ApplicationConfig object."""
        # Handle enum conversions
        if "environment" in config_dict:
            config_dict["environment"] = Environment(config_dict["environment"])
        
        if "llm" in config_dict:
            llm_dict = config_dict["llm"]
            if "provider" in llm_dict:
                llm_dict["provider"] = LLMProvider(llm_dict["provider"])
            config_dict["llm"] = LLMConfig(**llm_dict)
        
        if "chunking" in config_dict:
            chunking_dict = config_dict["chunking"]
            if "strategy" in chunking_dict:
                chunking_dict["strategy"] = ChunkingStrategy(chunking_dict["strategy"])
            config_dict["chunking"] = ChunkingConfig(**chunking_dict)
        
        if "processing" in config_dict:
            processing_dict = config_dict["processing"]
            if "mode" in processing_dict:
                processing_dict["mode"] = ProcessingMode(processing_dict["mode"])
            config_dict["processing"] = ProcessingConfig(**processing_dict)
        
        # Handle other nested configurations
        for key, config_class in [
            ("monitoring", MonitoringConfig),
            ("security", SecurityConfig),
            ("storage", StorageConfig),
            ("database", DatabaseConfig)
        ]:
            if key in config_dict:
                config_dict[key] = config_class(**config_dict[key])
        
        return ApplicationConfig(**config_dict)
    
    def save_config(self, config: ApplicationConfig, file_path: Optional[Path] = None):
        """Save configuration to file."""
        if not file_path:
            file_path = self.config_file
        
        config_dict = config.to_dict()
        
        # Convert enums to strings for serialization
        config_dict = self._prepare_for_serialization(config_dict)
        
        with open(file_path, 'w') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif file_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2, default=str)
        
        self.logger.info(f"Configuration saved to {file_path}")
    
    def _prepare_for_serialization(self, obj: Any) -> Any:
        """Prepare object for serialization by converting enums and paths."""
        if isinstance(obj, dict):
            return {k: self._prepare_for_serialization(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_serialization(item) for item in obj]
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
    
    def get_config(self) -> ApplicationConfig:
        """Get current configuration."""
        if not self.config:
            self.config = self.load_config()
        return self.config
    
    def reload_config(self) -> ApplicationConfig:
        """Reload configuration from file."""
        self.config = None
        return self.load_config()


# Global configuration instance
_config_manager = ConfigManager()


def get_config() -> ApplicationConfig:
    """Get global application configuration."""
    return _config_manager.get_config()


def load_config(config_file: Optional[Path] = None) -> ApplicationConfig:
    """Load configuration from file."""
    return _config_manager.load_config(config_file)


def reload_config() -> ApplicationConfig:
    """Reload configuration from file."""
    return _config_manager.reload_config()