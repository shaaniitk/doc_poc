#!/usr/bin/env python3
"""
Test orchestrator initialization.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Testing orchestrator initialization...")

try:
    from core.langgraph_orchestrator import LangGraphOrchestrator
    from core.state_manager import CentralizedStateManager
    
    # Test orchestrator initialization
    config = {
        'llm_provider': 'mistral',
        'model_name': 'mistral-large-latest',
        'api_key': 'test_key',
        'max_tokens': 4000,
        'temperature': 0.1
    }
    
    print("Creating orchestrator...")
    orchestrator = LangGraphOrchestrator(config)
    print("✅ Orchestrator initialized successfully")
    
    print("Creating state manager...")
    state_manager = CentralizedStateManager()
    print("✅ State manager initialized successfully")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print("Orchestrator test completed.")