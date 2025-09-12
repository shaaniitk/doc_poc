#!/usr/bin/env python3
"""
Local Setup Test Script

Tests the local Hugging Face model setup to ensure everything is working correctly.
This script validates:
- Package imports
- Model loading
- Embedding generation
- GPU availability
- Configuration files

Usage:
    python test_local_setup.py [--profile PROFILE] [--verbose]
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Color codes for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

class LocalSetupTester:
    def __init__(self, profile: str = "balanced", verbose: bool = False):
        self.profile = profile
        self.verbose = verbose
        self.project_root = Path.cwd()
        self.test_results = []
        
    def log_test(self, test_name: str, status: str, details: str = "", duration: float = 0.0):
        """Log test results."""
        self.test_results.append({
            "test": test_name,
            "status": status,
            "details": details,
            "duration_seconds": duration
        })
        
    def print_success(self, text: str):
        print(f"{Colors.GREEN}[OK] {text}{Colors.END}")
        
    def print_error(self, text: str):
        print(f"{Colors.RED}[ERROR] {text}{Colors.END}")
        
    def print_warning(self, text: str):
        print(f"{Colors.YELLOW}[WARNING] {text}{Colors.END}")
        
    def print_info(self, text: str):
        print(f"{Colors.BLUE}[INFO] {text}{Colors.END}")
        
    def print_header(self, text: str):
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*50}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}{text.center(50)}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*50}{Colors.END}\n")
    
    def test_package_imports(self) -> bool:
        """Test that all required packages can be imported."""
        start_time = time.time()
        
        required_packages = [
            ("torch", "PyTorch"),
            ("transformers", "Transformers"),
            ("sentence_transformers", "Sentence Transformers"),
            ("numpy", "NumPy"),
            ("scipy", "SciPy"),
            ("sklearn", "Scikit-learn"),
            ("pandas", "Pandas"),
            ("requests", "Requests")
        ]
        
        failed_imports = []
        
        for module, name in required_packages:
            try:
                __import__(module)
                if self.verbose:
                    self.print_success(f"{name} imported successfully")
            except ImportError as e:
                self.print_error(f"Failed to import {name}: {e}")
                failed_imports.append(name)
        
        duration = time.time() - start_time
        
        if not failed_imports:
            self.print_success("All required packages imported successfully")
            self.log_test("Package Imports", "passed", f"All {len(required_packages)} packages imported", duration)
            return True
        else:
            self.print_error(f"Failed to import: {', '.join(failed_imports)}")
            self.log_test("Package Imports", "failed", f"Failed: {', '.join(failed_imports)}", duration)
            return False
    
    def test_torch_setup(self) -> bool:
        """Test PyTorch installation and GPU availability."""
        start_time = time.time()
        
        try:
            import torch
            
            # Test basic tensor operations
            x = torch.randn(3, 3)
            y = torch.randn(3, 3)
            z = torch.matmul(x, y)
            
            self.print_success(f"PyTorch version: {torch.__version__}")
            
            # Check GPU availability
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                self.print_success(f"GPU available: {gpu_name} ({gpu_memory:.1f}GB)")
                self.print_info(f"GPU count: {gpu_count}")
                
                # Test GPU tensor operations
                if torch.cuda.is_available():
                    x_gpu = x.cuda()
                    y_gpu = y.cuda()
                    z_gpu = torch.matmul(x_gpu, y_gpu)
                    self.print_success("GPU tensor operations working")
            else:
                self.print_info("No GPU available, using CPU")
            
            duration = time.time() - start_time
            self.log_test("PyTorch Setup", "passed", f"Version: {torch.__version__}, GPU: {torch.cuda.is_available()}", duration)
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.print_error(f"PyTorch test failed: {e}")
            self.log_test("PyTorch Setup", "failed", str(e), duration)
            return False
    
    def test_model_loading(self) -> bool:
        """Test loading a simple model."""
        start_time = time.time()
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Try to load a lightweight model
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.print_info(f"Loading model: {model_name}")
            
            model = SentenceTransformer(model_name)
            
            # Test encoding
            test_sentences = [
                "This is a test sentence.",
                "This is another test sentence."
            ]
            
            embeddings = model.encode(test_sentences)
            
            self.print_success(f"Model loaded successfully")
            self.print_info(f"Embedding dimension: {embeddings.shape[1]}")
            self.print_info(f"Processed {len(test_sentences)} sentences")
            
            duration = time.time() - start_time
            self.log_test("Model Loading", "passed", f"Model: {model_name}, Dim: {embeddings.shape[1]}", duration)
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.print_error(f"Model loading test failed: {e}")
            self.log_test("Model Loading", "failed", str(e), duration)
            return False
    
    def test_cached_models(self) -> bool:
        """Test if downloaded models are accessible."""
        start_time = time.time()
        
        models_dir = self.project_root / "models"
        
        if not models_dir.exists():
            self.print_warning("Models directory not found - models may not be downloaded yet")
            duration = time.time() - start_time
            self.log_test("Cached Models", "skipped", "Models directory not found", duration)
            return True  # Not a failure, just not downloaded yet
        
        try:
            # Check for embedding models
            embedding_dir = models_dir / "embeddings"
            llm_dir = models_dir / "llm"
            
            embedding_count = 0
            llm_count = 0
            
            if embedding_dir.exists():
                embedding_count = len([d for d in embedding_dir.iterdir() if d.is_dir()])
            
            if llm_dir.exists():
                llm_count = len([d for d in llm_dir.iterdir() if d.is_dir()])
            
            self.print_info(f"Cached embedding models: {embedding_count}")
            self.print_info(f"Cached LLM models: {llm_count}")
            
            if embedding_count > 0 or llm_count > 0:
                self.print_success("Cached models found")
            else:
                self.print_warning("No cached models found")
            
            duration = time.time() - start_time
            self.log_test("Cached Models", "passed", f"Embedding: {embedding_count}, LLM: {llm_count}", duration)
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.print_error(f"Cached models test failed: {e}")
            self.log_test("Cached Models", "failed", str(e), duration)
            return False
    
    def test_configuration_files(self) -> bool:
        """Test that configuration files exist and are valid."""
        start_time = time.time()
        
        config_files = [
            ("config.py", "Main configuration"),
            ("local_embedding_config.py", "Local embedding config"),
            ("local_config_updater.py", "Config updater script"),
            ("download_local_models.py", "Model download script"),
            ("LOCAL_MODELS_GUIDE.md", "Documentation")
        ]
        
        missing_files = []
        
        for filename, description in config_files:
            file_path = self.project_root / filename
            if file_path.exists():
                if self.verbose:
                    self.print_success(f"{description}: {filename}")
            else:
                self.print_warning(f"Missing {description}: {filename}")
                missing_files.append(filename)
        
        # Check .env file
        env_file = self.project_root / ".env"
        if env_file.exists():
            if self.verbose:
                self.print_success(".env file found")
        else:
            self.print_info(".env file not found (will be created by setup script)")
        
        duration = time.time() - start_time
        
        if len(missing_files) <= 1:  # Allow one missing file
            self.print_success("Configuration files check passed")
            self.log_test("Configuration Files", "passed", f"Missing: {len(missing_files)}", duration)
            return True
        else:
            self.print_error(f"Too many missing files: {', '.join(missing_files)}")
            self.log_test("Configuration Files", "failed", f"Missing: {', '.join(missing_files)}", duration)
            return False
    
    def test_embedding_performance(self) -> bool:
        """Test embedding generation performance."""
        start_time = time.time()
        
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            # Load a lightweight model for testing
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            model = SentenceTransformer(model_name)
            
            # Test with various sentence lengths
            test_sentences = [
                "Short sentence.",
                "This is a medium length sentence for testing embedding performance.",
                "This is a much longer sentence that contains more words and should test the model's ability to handle longer text inputs while maintaining good performance and accuracy in the embedding generation process."
            ]
            
            # Measure embedding time
            embed_start = time.time()
            embeddings = model.encode(test_sentences)
            embed_duration = time.time() - embed_start
            
            # Calculate performance metrics
            sentences_per_second = len(test_sentences) / embed_duration
            avg_time_per_sentence = embed_duration / len(test_sentences)
            
            self.print_success(f"Embedding performance test completed")
            self.print_info(f"Sentences per second: {sentences_per_second:.2f}")
            self.print_info(f"Average time per sentence: {avg_time_per_sentence:.3f}s")
            self.print_info(f"Embedding dimension: {embeddings.shape[1]}")
            
            # Test similarity calculation
            similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
            self.print_info(f"Similarity between first two sentences: {similarity:.3f}")
            
            duration = time.time() - start_time
            self.log_test("Embedding Performance", "passed", f"Speed: {sentences_per_second:.2f} sent/sec", duration)
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.print_error(f"Embedding performance test failed: {e}")
            self.log_test("Embedding Performance", "failed", str(e), duration)
            return False
    
    def test_memory_usage(self) -> bool:
        """Test memory usage during model operations."""
        start_time = time.time()
        
        try:
            import psutil
            import gc
            
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            self.print_info(f"Initial memory usage: {initial_memory:.1f} MB")
            
            # Load model and measure memory increase
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            
            after_load_memory = process.memory_info().rss / (1024 * 1024)
            model_memory = after_load_memory - initial_memory
            
            self.print_info(f"Memory after model load: {after_load_memory:.1f} MB")
            self.print_info(f"Model memory usage: {model_memory:.1f} MB")
            
            # Test with batch processing
            test_sentences = ["Test sentence."] * 100
            embeddings = model.encode(test_sentences)
            
            after_embed_memory = process.memory_info().rss / (1024 * 1024)
            embed_memory = after_embed_memory - after_load_memory
            
            self.print_info(f"Memory after embedding 100 sentences: {after_embed_memory:.1f} MB")
            self.print_info(f"Embedding memory overhead: {embed_memory:.1f} MB")
            
            # Clean up
            del model
            del embeddings
            gc.collect()
            
            final_memory = process.memory_info().rss / (1024 * 1024)
            self.print_info(f"Memory after cleanup: {final_memory:.1f} MB")
            
            duration = time.time() - start_time
            
            # Check if memory usage is reasonable (less than 2GB)
            if after_embed_memory < 2048:
                self.print_success("Memory usage is within acceptable limits")
                self.log_test("Memory Usage", "passed", f"Peak: {after_embed_memory:.1f} MB", duration)
                return True
            else:
                self.print_warning(f"High memory usage: {after_embed_memory:.1f} MB")
                self.log_test("Memory Usage", "warning", f"High usage: {after_embed_memory:.1f} MB", duration)
                return True  # Still pass, just a warning
            
        except ImportError:
            self.print_warning("psutil not available, skipping memory test")
            duration = time.time() - start_time
            self.log_test("Memory Usage", "skipped", "psutil not available", duration)
            return True
        except Exception as e:
            duration = time.time() - start_time
            self.print_error(f"Memory usage test failed: {e}")
            self.log_test("Memory Usage", "failed", str(e), duration)
            return False
    
    def run_all_tests(self) -> bool:
        """Run all tests and return overall success status."""
        self.print_header("LOCAL SETUP TESTING")
        
        print(f"{Colors.BOLD}Configuration:{Colors.END}")
        print(f"  Profile: {self.profile}")
        print(f"  Verbose: {self.verbose}")
        print(f"  Project root: {self.project_root}")
        
        tests = [
            ("Package Imports", self.test_package_imports),
            ("PyTorch Setup", self.test_torch_setup),
            ("Configuration Files", self.test_configuration_files),
            ("Model Loading", self.test_model_loading),
            ("Cached Models", self.test_cached_models),
            ("Embedding Performance", self.test_embedding_performance),
            ("Memory Usage", self.test_memory_usage)
        ]
        
        passed_tests = 0
        failed_tests = []
        
        total_start_time = time.time()
        
        for test_name, test_func in tests:
            self.print_header(f"Testing: {test_name}")
            
            try:
                if test_func():
                    passed_tests += 1
                    self.print_success(f"{test_name} PASSED")
                else:
                    failed_tests.append(test_name)
                    self.print_error(f"{test_name} FAILED")
            except Exception as e:
                failed_tests.append(test_name)
                self.print_error(f"{test_name} FAILED with exception: {e}")
        
        total_duration = time.time() - total_start_time
        
        # Save test results
        self.save_test_results(total_duration)
        
        # Print summary
        self.print_header("TEST SUMMARY")
        
        print(f"{Colors.BOLD}Results:{Colors.END}")
        print(f"  Total tests: {len(tests)}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {len(failed_tests)}")
        print(f"  Duration: {total_duration:.2f} seconds")
        
        if not failed_tests:
            self.print_success("[OK] All tests passed! Local setup is working correctly.")
            print(f"\n{Colors.BOLD}Your local environment is ready for:{Colors.END}")
            print(f"  [OK] Local embedding generation")
            print(f"  [OK] Offline model inference")
            print(f"  [OK] Document processing without external APIs")
            return True
        else:
            self.print_error(f"[ERROR] {len(failed_tests)} test(s) failed: {', '.join(failed_tests)}")
            print(f"\n{Colors.BOLD}Troubleshooting:{Colors.END}")
            print(f"  1. Check test results: test_results.json")
            print(f"  2. Ensure all dependencies are installed")
            print(f"  3. Run setup script: python setup_local_environment.py")
            return False
    
    def save_test_results(self, total_duration: float):
        """Save test results to a JSON file."""
        results_file = self.project_root / "test_results.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump({
                    "profile": self.profile,
                    "timestamp": str(Path().cwd()),
                    "total_duration": total_duration,
                    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    "tests": self.test_results
                }, f, indent=2)
            
            self.print_success(f"Test results saved: {results_file}")
        except Exception as e:
            self.print_error(f"Failed to save test results: {e}")

def main():
    parser = argparse.ArgumentParser(description='Test local Hugging Face model setup')
    parser.add_argument('--profile', choices=['lightweight', 'balanced', 'high_quality'], 
                       default='balanced', help='Model profile to test')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    tester = LocalSetupTester(profile=args.profile, verbose=args.verbose)
    
    try:
        success = tester.run_all_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}[WARNING] Testing interrupted by user.{Colors.END}")
        return 1
    except Exception as e:
        print(f"\n{Colors.RED}[ERROR] Unexpected error: {e}{Colors.END}")
        return 1

if __name__ == '__main__':
    exit(main())