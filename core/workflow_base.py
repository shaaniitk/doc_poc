"""Base classes for workflow nodes and configuration."""

import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .models import WorkflowState
from .state_manager import CentralizedStateManager

logger = logging.getLogger(__name__)

@dataclass
class NodeConfig:
    """Configuration for workflow nodes."""
    name: str
    parallel_execution: bool = False
    max_retries: int = 3
    timeout_seconds: float = 60.0
    required_inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    error_recovery_strategy: str = "retry"  # "retry", "skip", "fail"
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseWorkflowNode(ABC):
    """Base class for all workflow nodes."""
    
    def __init__(self, config: NodeConfig, state_manager: CentralizedStateManager):
        self.config = config
        self.state_manager = state_manager
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._execution_count = 0
        self._last_execution_time = None
        
    async def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute the node with error handling and retries."""
        self._execution_count += 1
        self._last_execution_time = datetime.now(timezone.utc)
        
        for attempt in range(self.config.max_retries + 1):
            try:
                self.logger.info(f"Executing {self.config.name} (attempt {attempt + 1})")
                
                # Validate inputs
                self._validate_inputs(state)
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    self._execute_impl(state),
                    timeout=self.config.timeout_seconds
                )
                
                # Validate outputs
                self._validate_outputs(result)
                
                self.logger.info(f"Successfully executed {self.config.name}")
                return result
                
            except asyncio.TimeoutError:
                self.logger.error(f"Timeout in {self.config.name} after {self.config.timeout_seconds}s")
                if attempt == self.config.max_retries:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                self.logger.error(f"Error in {self.config.name} (attempt {attempt + 1}): {e}")
                
                if attempt == self.config.max_retries:
                    if self.config.error_recovery_strategy == "skip":
                        self.logger.warning(f"Skipping {self.config.name} due to errors")
                        return state  # Return unchanged state
                    else:
                        raise
                
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return state
    
    @abstractmethod
    async def _execute_impl(self, state: WorkflowState) -> WorkflowState:
        """Implement the actual node logic."""
        pass
    
    def _validate_inputs(self, state: WorkflowState) -> None:
        """Validate that required inputs are present in the state."""
        print(f"DEBUG: Validating inputs for node {self.config.name}")
        print(f"DEBUG: State type: {type(state)}")
        print(f"DEBUG: State attributes: {[attr for attr in dir(state) if not attr.startswith('_')]}")
        print(f"DEBUG: Required inputs: {self.config.required_inputs}")
        print(f"DEBUG: Full state content for {self.config.name}: {list(state.keys()) if isinstance(state, dict) else 'Not a dict'}")
        
        for required_input in self.config.required_inputs:
            # Check if it's a dictionary (which it should be)
            if isinstance(state, dict):
                print(f"DEBUG: Checking '{required_input}' for {self.config.name}: exists={required_input in state}")
                if required_input not in state or state[required_input] is None:
                    raise ValueError(f"Required input '{required_input}' not found in state")
            else:
                # Fallback for object-style access
                has_attr = hasattr(state, required_input)
                attr_value = getattr(state, required_input, None) if has_attr else None
                print(f"DEBUG: Checking '{required_input}': has_attr={has_attr}, value={attr_value}")
                
                if not has_attr or attr_value is None:
                    raise ValueError(f"Required input '{required_input}' not found in state")
    
    def _validate_outputs(self, state: WorkflowState) -> None:
        """Validate that expected outputs are present in the state."""
        for expected_output in self.config.outputs:
            # Handle both dict and object-style state access
            if isinstance(state, dict):
                exists = expected_output in state
            else:
                exists = hasattr(state, expected_output)
            
            if not exists:
                self.logger.warning(f"Expected output '{expected_output}' not found in state")
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics for this node."""
        return {
            "name": self.config.name,
            "execution_count": self._execution_count,
            "last_execution_time": self._last_execution_time.isoformat() if self._last_execution_time else None,
            "max_retries": self.config.max_retries,
            "timeout_seconds": self.config.timeout_seconds
        }