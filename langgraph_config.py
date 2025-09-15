"""Configuration and session management for LangGraph workflow"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
import os
from enum import Enum
import uuid


class ConfigurationLevel(Enum):
    """Configuration hierarchy levels"""
    SYSTEM = "system"          # System-wide defaults
    PROJECT = "project"        # Project-specific settings
    SESSION = "session"        # Session-specific overrides
    RUNTIME = "runtime"        # Runtime parameter overrides


class SessionStatus(Enum):
    """Session status tracking"""
    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class ProcessingConfiguration:
    """Configuration for document processing pipeline"""
    
    # Document processing settings
    chunk_size: int = 1000
    overlap_size: int = 200
    max_document_size: int = 50 * 1024 * 1024  # 50MB
    supported_formats: List[str] = field(default_factory=lambda: [
        'pdf', 'docx', 'txt', 'md', 'html'
    ])
    
    # Knowledge graph settings
    kg_extraction_method: str = "hybrid"  # "rule_based", "ml_based", "hybrid"
    kg_confidence_threshold: float = 0.7
    max_nodes_per_chunk: int = 10
    max_edges_per_node: int = 5
    
    # Semantic mapping settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.75
    max_mappings_per_chunk: int = 5
    use_semantic_clustering: bool = True
    
    # LLM processing settings
    llm_model: str = "gpt-4"
    max_tokens: int = 2000
    temperature: float = 0.3
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # Output settings
    output_formats: List[str] = field(default_factory=lambda: ['json', 'markdown'])
    output_dir: str = "./output"
    include_metadata: bool = True
    include_analytics: bool = True
    
    # Performance settings
    max_concurrent_chunks: int = 5
    timeout_seconds: int = 300
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Feature flags
    use_enhanced_processing: bool = True
    enable_caching: bool = True
    enable_analytics: bool = True
    enable_error_recovery: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'chunk_size': self.chunk_size,
            'overlap_size': self.overlap_size,
            'max_document_size': self.max_document_size,
            'supported_formats': self.supported_formats,
            'kg_extraction_method': self.kg_extraction_method,
            'kg_confidence_threshold': self.kg_confidence_threshold,
            'max_nodes_per_chunk': self.max_nodes_per_chunk,
            'max_edges_per_node': self.max_edges_per_node,
            'embedding_model': self.embedding_model,
            'similarity_threshold': self.similarity_threshold,
            'max_mappings_per_chunk': self.max_mappings_per_chunk,
            'use_semantic_clustering': self.use_semantic_clustering,
            'llm_model': self.llm_model,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty,
            'output_formats': self.output_formats,
            'output_dir': self.output_dir,
            'include_metadata': self.include_metadata,
            'include_analytics': self.include_analytics,
            'max_concurrent_chunks': self.max_concurrent_chunks,
            'timeout_seconds': self.timeout_seconds,
            'retry_attempts': self.retry_attempts,
            'retry_delay': self.retry_delay,
            'use_enhanced_processing': self.use_enhanced_processing,
            'enable_caching': self.enable_caching,
            'enable_analytics': self.enable_analytics,
            'enable_error_recovery': self.enable_error_recovery
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingConfiguration':
        """Create configuration from dictionary"""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
    
    def merge_with(self, other: 'ProcessingConfiguration') -> 'ProcessingConfiguration':
        """Merge this configuration with another, other takes precedence"""
        merged_data = self.to_dict()
        merged_data.update(other.to_dict())
        return ProcessingConfiguration.from_dict(merged_data)
    
    def validate(self) -> List[str]:
        """Validate configuration parameters"""
        errors = []
        
        if self.chunk_size <= 0:
            errors.append("chunk_size must be positive")
        
        if self.overlap_size < 0:
            errors.append("overlap_size cannot be negative")
        
        if self.overlap_size >= self.chunk_size:
            errors.append("overlap_size must be less than chunk_size")
        
        if not 0 <= self.kg_confidence_threshold <= 1:
            errors.append("kg_confidence_threshold must be between 0 and 1")
        
        if not 0 <= self.similarity_threshold <= 1:
            errors.append("similarity_threshold must be between 0 and 1")
        
        if not 0 <= self.temperature <= 2:
            errors.append("temperature must be between 0 and 2")
        
        if not 0 <= self.top_p <= 1:
            errors.append("top_p must be between 0 and 1")
        
        if self.max_tokens <= 0:
            errors.append("max_tokens must be positive")
        
        if self.timeout_seconds <= 0:
            errors.append("timeout_seconds must be positive")
        
        if not self.output_formats:
            errors.append("output_formats cannot be empty")
        
        return errors


@dataclass
class SessionInfo:
    """Session information and tracking"""
    
    session_id: str
    created_at: datetime
    status: SessionStatus = SessionStatus.CREATED
    last_activity: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Session metadata
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    session_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Processing context
    document_path: Optional[str] = None
    configuration_level: ConfigurationLevel = ConfigurationLevel.SESSION
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    # Session statistics
    total_processing_time: float = 0.0
    documents_processed: int = 0
    errors_encountered: int = 0
    warnings_encountered: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session info to dictionary"""
        return {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'status': self.status.value,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'user_id': self.user_id,
            'project_id': self.project_id,
            'session_name': self.session_name,
            'tags': self.tags,
            'document_path': self.document_path,
            'configuration_level': self.configuration_level.value,
            'custom_config': self.custom_config,
            'total_processing_time': self.total_processing_time,
            'documents_processed': self.documents_processed,
            'errors_encountered': self.errors_encountered,
            'warnings_encountered': self.warnings_encountered
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionInfo':
        """Create session info from dictionary"""
        session_info = cls(
            session_id=data['session_id'],
            created_at=datetime.fromisoformat(data['created_at']),
            status=SessionStatus(data.get('status', SessionStatus.CREATED.value)),
            last_activity=datetime.fromisoformat(data['last_activity']) if data.get('last_activity') else None,
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            user_id=data.get('user_id'),
            project_id=data.get('project_id'),
            session_name=data.get('session_name'),
            tags=data.get('tags', []),
            document_path=data.get('document_path'),
            configuration_level=ConfigurationLevel(data.get('configuration_level', ConfigurationLevel.SESSION.value)),
            custom_config=data.get('custom_config', {}),
            total_processing_time=data.get('total_processing_time', 0.0),
            documents_processed=data.get('documents_processed', 0),
            errors_encountered=data.get('errors_encountered', 0),
            warnings_encountered=data.get('warnings_encountered', 0)
        )
        return session_info
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
    
    def is_expired(self) -> bool:
        """Check if session has expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def extend_expiration(self, hours: int = 24):
        """Extend session expiration"""
        self.expires_at = datetime.now() + timedelta(hours=hours)


class ConfigurationManager:
    """Manages configuration hierarchy and merging"""
    
    def __init__(self, config_dir: str = "./config"):
        self.config_dir = config_dir
        self._configurations: Dict[ConfigurationLevel, ProcessingConfiguration] = {}
        self._load_configurations()
    
    def _load_configurations(self):
        """Load configurations from files"""
        # Load system configuration
        system_config_path = os.path.join(self.config_dir, "system_config.json")
        if os.path.exists(system_config_path):
            with open(system_config_path, 'r') as f:
                system_data = json.load(f)
                self._configurations[ConfigurationLevel.SYSTEM] = ProcessingConfiguration.from_dict(system_data)
        else:
            self._configurations[ConfigurationLevel.SYSTEM] = ProcessingConfiguration()
        
        # Load project configuration
        project_config_path = os.path.join(self.config_dir, "project_config.json")
        if os.path.exists(project_config_path):
            with open(project_config_path, 'r') as f:
                project_data = json.load(f)
                self._configurations[ConfigurationLevel.PROJECT] = ProcessingConfiguration.from_dict(project_data)
    
    def get_configuration(self, 
                         session_config: Optional[ProcessingConfiguration] = None,
                         runtime_overrides: Optional[Dict[str, Any]] = None) -> ProcessingConfiguration:
        """Get merged configuration based on hierarchy"""
        # Start with system configuration
        config = self._configurations.get(ConfigurationLevel.SYSTEM, ProcessingConfiguration())
        
        # Merge project configuration
        if ConfigurationLevel.PROJECT in self._configurations:
            config = config.merge_with(self._configurations[ConfigurationLevel.PROJECT])
        
        # Merge session configuration
        if session_config:
            config = config.merge_with(session_config)
        
        # Apply runtime overrides
        if runtime_overrides:
            runtime_config = ProcessingConfiguration.from_dict(runtime_overrides)
            config = config.merge_with(runtime_config)
        
        return config
    
    def save_configuration(self, config: ProcessingConfiguration, level: ConfigurationLevel):
        """Save configuration to file"""
        os.makedirs(self.config_dir, exist_ok=True)
        
        filename_map = {
            ConfigurationLevel.SYSTEM: "system_config.json",
            ConfigurationLevel.PROJECT: "project_config.json"
        }
        
        if level in filename_map:
            config_path = os.path.join(self.config_dir, filename_map[level])
            with open(config_path, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            
            self._configurations[level] = config
    
    def validate_configuration(self, config: ProcessingConfiguration) -> List[str]:
        """Validate configuration"""
        return config.validate()


class SessionManager:
    """Manages session lifecycle and persistence"""
    
    def __init__(self, session_dir: str = "./sessions"):
        self.session_dir = session_dir
        self._active_sessions: Dict[str, SessionInfo] = {}
        os.makedirs(session_dir, exist_ok=True)
    
    def create_session(self, 
                      user_id: Optional[str] = None,
                      project_id: Optional[str] = None,
                      session_name: Optional[str] = None,
                      document_path: Optional[str] = None,
                      expiration_hours: int = 24,
                      custom_config: Optional[Dict[str, Any]] = None) -> SessionInfo:
        """Create a new session"""
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        session_info = SessionInfo(
            session_id=session_id,
            created_at=now,
            status=SessionStatus.CREATED,
            last_activity=now,
            expires_at=now + timedelta(hours=expiration_hours),
            user_id=user_id,
            project_id=project_id,
            session_name=session_name or f"Session-{session_id[:8]}",
            document_path=document_path,
            custom_config=custom_config or {}
        )
        
        self._active_sessions[session_id] = session_info
        self._save_session(session_info)
        
        return session_info
    
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session by ID"""
        if session_id in self._active_sessions:
            session = self._active_sessions[session_id]
            if session.is_expired():
                session.status = SessionStatus.EXPIRED
                self._save_session(session)
                return None
            return session
        
        # Try to load from disk
        session = self._load_session(session_id)
        if session and not session.is_expired():
            self._active_sessions[session_id] = session
            return session
        
        return None
    
    def update_session(self, session_info: SessionInfo):
        """Update session information"""
        session_info.update_activity()
        self._active_sessions[session_info.session_id] = session_info
        self._save_session(session_info)
    
    def end_session(self, session_id: str, status: SessionStatus = SessionStatus.COMPLETED):
        """End a session"""
        if session_id in self._active_sessions:
            session = self._active_sessions[session_id]
            session.status = status
            session.update_activity()
            self._save_session(session)
            del self._active_sessions[session_id]
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        expired_sessions = []
        
        for session_id, session in self._active_sessions.items():
            if session.is_expired():
                session.status = SessionStatus.EXPIRED
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.end_session(session_id, SessionStatus.EXPIRED)
    
    def list_sessions(self, 
                     user_id: Optional[str] = None,
                     project_id: Optional[str] = None,
                     status: Optional[SessionStatus] = None) -> List[SessionInfo]:
        """List sessions with optional filtering"""
        sessions = []
        
        # Check active sessions
        for session in self._active_sessions.values():
            if self._matches_filter(session, user_id, project_id, status):
                sessions.append(session)
        
        # Check persisted sessions
        for filename in os.listdir(self.session_dir):
            if filename.endswith('.json'):
                session_id = filename[:-5]  # Remove .json extension
                if session_id not in self._active_sessions:
                    session = self._load_session(session_id)
                    if session and self._matches_filter(session, user_id, project_id, status):
                        sessions.append(session)
        
        return sorted(sessions, key=lambda s: s.created_at, reverse=True)
    
    def _matches_filter(self, session: SessionInfo, 
                       user_id: Optional[str],
                       project_id: Optional[str],
                       status: Optional[SessionStatus]) -> bool:
        """Check if session matches filter criteria"""
        if user_id and session.user_id != user_id:
            return False
        if project_id and session.project_id != project_id:
            return False
        if status and session.status != status:
            return False
        return True
    
    def _save_session(self, session_info: SessionInfo):
        """Save session to disk"""
        session_path = os.path.join(self.session_dir, f"{session_info.session_id}.json")
        with open(session_path, 'w') as f:
            json.dump(session_info.to_dict(), f, indent=2)
    
    def _load_session(self, session_id: str) -> Optional[SessionInfo]:
        """Load session from disk"""
        session_path = os.path.join(self.session_dir, f"{session_id}.json")
        if os.path.exists(session_path):
            try:
                with open(session_path, 'r') as f:
                    session_data = json.load(f)
                    return SessionInfo.from_dict(session_data)
            except (json.JSONDecodeError, KeyError, ValueError):
                # Handle corrupted session files
                pass
        return None


# Global instances
_config_manager = None
_session_manager = None


def get_config_manager() -> ConfigurationManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager


def get_session_manager() -> SessionManager:
    """Get global session manager instance"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def create_session_with_config(document_path: str,
                              user_id: Optional[str] = None,
                              project_id: Optional[str] = None,
                              custom_config: Optional[Dict[str, Any]] = None) -> tuple[SessionInfo, ProcessingConfiguration]:
    """Create session and get merged configuration"""
    session_manager = get_session_manager()
    config_manager = get_config_manager()
    
    # Create session
    session_info = session_manager.create_session(
        user_id=user_id,
        project_id=project_id,
        document_path=document_path,
        custom_config=custom_config
    )
    
    # Get merged configuration
    session_config = ProcessingConfiguration.from_dict(custom_config) if custom_config else None
    final_config = config_manager.get_configuration(session_config=session_config)
    
    return session_info, final_config