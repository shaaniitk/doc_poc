#!/usr/bin/env python3
"""
Simple test to isolate import issues.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Testing imports...")

try:
    from core.langgraph_orchestrator import LangGraphOrchestrator
    print("✅ LangGraphOrchestrator imported successfully")
except Exception as e:
    print(f"❌ Failed to import LangGraphOrchestrator: {e}")
    import traceback
    traceback.print_exc()

try:
    from core.state_manager import CentralizedStateManager
    print("✅ CentralizedStateManager imported successfully")
except Exception as e:
    print(f"❌ Failed to import CentralizedStateManager: {e}")
    import traceback
    traceback.print_exc()

try:
    from core.models import WorkflowState, DocumentMetadata, DocumentFormat
    print("✅ Models imported successfully")
except Exception as e:
    print(f"❌ Failed to import models: {e}")
    import traceback
    traceback.print_exc()

print("Import test completed.")