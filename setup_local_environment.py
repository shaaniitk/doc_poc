#!/usr/bin/env python3
"""
Local Environment Setup Script

Automated setup for running the document processing system with local Hugging Face models.
This script handles:
- Environment validation
- Dependency installation
- Model downloading
- Configuration updates
- Testing

Usage:
    python setup_local_environment.py [--profile PROFILE] [--skip-models] [--skip-test]
    
    --profile: Model profile (lightweight, balanced, high_quality)
    --skip-models: Skip model downloading
    --skip-test: Skip final testing
    --force: Force reinstall/redownload everything
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

class LocalEnvironmentSetup:
    def __init__(self, profile: str = "balanced", skip_models: bool = False, 
                 skip_test: bool = False, force: bool = False):
        self.profile = profile
        self.skip_models = skip_models
        self.skip_test = skip_test
        self.force = force
        self.project_root = Path.cwd()
        self.setup_log = []
        
    def log_step(self, step: str, status: str, details: str = ""):
        """Log setup steps for debugging."""
        self.setup_log.append({
            "step": step,
            "status": status,
            "details": details
        })
        
    def print_header(self, text: str):
        """Print a formatted header."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{text.center(60)}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}\n")
        
    def print_success(self, text: str):
        """Print success message."""
        print(f"{Colors.GREEN}[OK] {text}{Colors.END}")
        
    def print_error(self, text: str):
        """Print error message."""
        print(f"{Colors.RED}[ERROR] {text}{Colors.END}")
        
    def print_warning(self, text: str):
        """Print warning message."""
        print(f"{Colors.YELLOW}[WARNING] {text}{Colors.END}")
        
    def print_info(self, text: str):
        """Print info message."""
        print(f"{Colors.BLUE}[INFO] {text}{Colors.END}")
        
    def run_command(self, cmd: List[str], description: str, 
                   capture_output: bool = True) -> Tuple[bool, str]:
        """Run a command and return success status and output."""
        try:
            self.print_info(f"Running: {description}")
            result = subprocess.run(
                cmd, 
                capture_output=capture_output, 
                text=True, 
                check=False
            )
            
            if result.returncode == 0:
                self.print_success(f"Completed: {description}")
                self.log_step(description, "success", result.stdout)
                return True, result.stdout
            else:
                self.print_error(f"Failed: {description}")
                self.print_error(f"Error: {result.stderr}")
                self.log_step(description, "failed", result.stderr)
                return False, result.stderr
                
        except Exception as e:
            self.print_error(f"Exception during {description}: {e}")
            self.log_step(description, "exception", str(e))
            return False, str(e)
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            self.print_success(f"Python version: {version.major}.{version.minor}.{version.micro}")
            return True
        else:
            self.print_error(f"Python 3.8+ required, found: {version.major}.{version.minor}.{version.micro}")
            return False
    
    def check_git_availability(self) -> bool:
        """Check if git is available for cloning repositories."""
        success, _ = self.run_command(["git", "--version"], "Checking Git availability")
        return success
    
    def check_disk_space(self, required_gb: float = 10.0) -> bool:
        """Check available disk space."""
        try:
            import shutil
            free_bytes = shutil.disk_usage(self.project_root).free
            free_gb = free_bytes / (1024**3)
            
            if free_gb >= required_gb:
                self.print_success(f"Disk space: {free_gb:.1f}GB available (required: {required_gb}GB)")
                return True
            else:
                self.print_error(f"Insufficient disk space: {free_gb:.1f}GB available (required: {required_gb}GB)")
                return False
        except Exception as e:
            self.print_warning(f"Could not check disk space: {e}")
            return True  # Proceed anyway
    
    def install_dependencies(self) -> bool:
        """Install required Python packages."""
        # Base requirements
        base_packages = [
            "torch",
            "transformers>=4.21.0",
            "sentence-transformers>=2.2.0",
            "numpy>=1.21.0",
            "scipy>=1.7.0",
            "scikit-learn>=1.0.0",
            "pandas>=1.3.0",
            "requests>=2.25.0",
            "tqdm>=4.62.0",
            "huggingface-hub>=0.10.0"
        ]
        
        # Check if requirements.txt exists and update it
        requirements_file = self.project_root / "requirements.txt"
        if requirements_file.exists():
            self.print_info("Updating existing requirements.txt")
            with open(requirements_file, 'r') as f:
                existing_reqs = f.read().splitlines()
            
            # Add new packages that aren't already present
            new_reqs = existing_reqs.copy()
            for package in base_packages:
                package_name = package.split('>=')[0].split('==')[0]
                if not any(package_name in req for req in existing_reqs):
                    new_reqs.append(package)
            
            with open(requirements_file, 'w') as f:
                f.write('\n'.join(new_reqs))
        else:
            self.print_info("Creating new requirements.txt")
            with open(requirements_file, 'w') as f:
                f.write('\n'.join(base_packages))
        
        # Install packages
        success, _ = self.run_command(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            "Installing Python dependencies"
        )
        
        if success:
            # Verify critical packages
            critical_imports = [
                ("torch", "PyTorch"),
                ("transformers", "Transformers"),
                ("sentence_transformers", "Sentence Transformers")
            ]
            
            for module, name in critical_imports:
                try:
                    __import__(module)
                    self.print_success(f"{name} imported successfully")
                except ImportError as e:
                    self.print_error(f"Failed to import {name}: {e}")
                    return False
        
        return success
    
    def download_models(self) -> bool:
        """Download models using the download script."""
        if self.skip_models:
            self.print_info("Skipping model download (--skip-models flag)")
            return True
        
        download_script = self.project_root / "download_local_models.py"
        if not download_script.exists():
            self.print_error("download_local_models.py not found")
            return False
        
        cmd = [sys.executable, str(download_script), "--profile", self.profile]
        if self.force:
            cmd.append("--force")
        
        success, _ = self.run_command(
            cmd,
            f"Downloading models for profile: {self.profile}",
            capture_output=False  # Show real-time output for downloads
        )
        
        return success
    
    def update_configuration(self) -> bool:
        """Update project configuration for local models."""
        config_updater = self.project_root / "local_config_updater.py"
        if not config_updater.exists():
            self.print_error("local_config_updater.py not found")
            return False
        
        cmd = [sys.executable, str(config_updater), "--backup", "--profile", self.profile]
        if self.force:
            cmd.append("--force")
        
        success, _ = self.run_command(
            cmd,
            "Updating configuration for local models"
        )
        
        return success
    
    def create_env_file(self) -> bool:
        """Create .env file for local configuration."""
        env_file = self.project_root / ".env"
        
        env_content = f"""# Local Model Configuration
# Generated by setup_local_environment.py

# Model Profile
MODEL_PROFILE={self.profile}

# Local Model Paths
MODEL_CACHE_DIR=./models
EMBEDDING_MODEL_PATH=./models/embeddings
LLM_MODEL_PATH=./models/llm

# Performance Settings
USE_GPU=auto
BATCH_SIZE=32
MAX_LENGTH=512

# API Settings (disabled for local mode)
OPENAI_API_KEY=disabled_local_mode
MISTRAL_API_KEY=disabled_local_mode

# Local Mode Flag
LOCAL_MODE=true
OFFLINE_MODE=true

# Logging
LOG_LEVEL=INFO
DEBUG_MODE=false
"""
        
        try:
            with open(env_file, 'w') as f:
                f.write(env_content)
            self.print_success(f"Created .env file: {env_file}")
            return True
        except Exception as e:
            self.print_error(f"Failed to create .env file: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Run basic tests to verify the setup."""
        if self.skip_test:
            self.print_info("Skipping tests (--skip-test flag)")
            return True
        
        # Test 1: Import test
        test_script = self.project_root / "test_local_setup.py"
        if test_script.exists():
            success, _ = self.run_command(
                [sys.executable, str(test_script)],
                "Running local setup tests"
            )
            if not success:
                return False
        else:
            self.print_warning("test_local_setup.py not found, skipping detailed tests")
        
        # Test 2: Basic import test
        test_code = """
import sys
try:
    import torch
    import transformers
    import sentence_transformers
    print("[OK] All critical packages imported successfully")
    
    # Test GPU availability
    if torch.cuda.is_available():
        print(f"[OK] GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] GPU not available, using CPU")
    
    print("[OK] Basic setup test passed")
    sys.exit(0)
except Exception as e:
    print(f"[ERROR] Import test failed: {e}")
    sys.exit(1)
"""
        
        success, _ = self.run_command(
            [sys.executable, "-c", test_code],
            "Running basic import test"
        )
        
        return success
    
    def save_setup_log(self) -> bool:
        """Save the setup log for debugging."""
        log_file = self.project_root / "setup_log.json"
        
        try:
            with open(log_file, 'w') as f:
                json.dump({
                    "profile": self.profile,
                    "timestamp": str(Path().cwd()),
                    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    "steps": self.setup_log
                }, f, indent=2)
            
            self.print_success(f"Setup log saved: {log_file}")
            return True
        except Exception as e:
            self.print_error(f"Failed to save setup log: {e}")
            return False
    
    def run_setup(self) -> bool:
        """Run the complete setup process."""
        self.print_header("LOCAL ENVIRONMENT SETUP")
        
        print(f"{Colors.BOLD}Configuration:{Colors.END}")
        print(f"  Profile: {self.profile}")
        print(f"  Skip models: {self.skip_models}")
        print(f"  Skip tests: {self.skip_test}")
        print(f"  Force reinstall: {self.force}")
        print(f"  Project root: {self.project_root}")
        
        steps = [
            ("System Requirements Check", self.check_system_requirements),
            ("Install Dependencies", self.install_dependencies),
            ("Download Models", self.download_models),
            ("Update Configuration", self.update_configuration),
            ("Create Environment File", self.create_env_file),
            ("Run Tests", self.run_tests),
            ("Save Setup Log", self.save_setup_log)
        ]
        
        failed_steps = []
        
        for step_name, step_func in steps:
            self.print_header(step_name)
            
            try:
                if step_func():
                    self.print_success(f"{step_name} completed successfully")
                else:
                    self.print_error(f"{step_name} failed")
                    failed_steps.append(step_name)
            except Exception as e:
                self.print_error(f"{step_name} failed with exception: {e}")
                failed_steps.append(step_name)
        
        # Final summary
        self.print_header("SETUP SUMMARY")
        
        if not failed_steps:
            self.print_success("Local environment setup completed successfully!")
            print(f"\n{Colors.BOLD}Next steps:{Colors.END}")
            print(f"1. Test the setup: python test_local_setup.py")
            print(f"2. Run your document processing with local models")
            print(f"3. Check the setup log for details: setup_log.json")
            return True
        else:
            self.print_error(f"Setup failed. Failed steps: {', '.join(failed_steps)}")
            print(f"\n{Colors.BOLD}Troubleshooting:{Colors.END}")
            print(f"1. Check the setup log: setup_log.json")
            print(f"2. Retry with --force flag")
            print(f"3. Install dependencies manually")
            return False
    
    def check_system_requirements(self) -> bool:
        """Check all system requirements."""
        checks = [
            ("Python Version", self.check_python_version),
            ("Git Availability", self.check_git_availability),
            ("Disk Space", lambda: self.check_disk_space(15.0))  # 15GB for safety
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            if not check_func():
                all_passed = False
        
        return all_passed

def main():
    parser = argparse.ArgumentParser(description='Set up local environment for document processing')
    parser.add_argument('--profile', choices=['lightweight', 'balanced', 'high_quality'], 
                       default='balanced', help='Model profile to use')
    parser.add_argument('--skip-models', action='store_true', help='Skip model downloading')
    parser.add_argument('--skip-test', action='store_true', help='Skip final testing')
    parser.add_argument('--force', action='store_true', help='Force reinstall/redownload everything')
    
    args = parser.parse_args()
    
    setup = LocalEnvironmentSetup(
        profile=args.profile,
        skip_models=args.skip_models,
        skip_test=args.skip_test,
        force=args.force
    )
    
    try:
        success = setup.run_setup()
        return 0 if success else 1
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}[WARNING] Setup interrupted by user.{Colors.END}")
        return 1
    except Exception as e:
        print(f"\n{Colors.RED}[ERROR] Unexpected error: {e}{Colors.END}")
        return 1

if __name__ == '__main__':
    exit(main())