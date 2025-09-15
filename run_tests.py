#!/usr/bin/env python3
"""Comprehensive test runner for the LangGraph document processing pipeline."""

import sys
import os
import asyncio
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import json

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text

console = Console()


class TestRunner:
    """Comprehensive test runner with reporting and analysis."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_results: Dict[str, Any] = {}
        self.start_time = None
        self.end_time = None
    
    def run_all_tests(self, 
                     test_types: Optional[List[str]] = None,
                     verbose: bool = False,
                     coverage: bool = False,
                     parallel: bool = False) -> bool:
        """Run all tests with specified options."""
        
        console.print(Panel.fit(
            "[bold blue]LangGraph Document Processing Pipeline - Test Suite[/bold blue]",
            border_style="blue"
        ))
        
        self.start_time = time.time()
        
        # Default test types
        if test_types is None:
            test_types = ['unit', 'integration', 'performance', 'config']
        
        all_passed = True
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            # Run each test type
            for test_type in test_types:
                task = progress.add_task(f"Running {test_type} tests...", total=1)
                
                success = self._run_test_type(
                    test_type, 
                    verbose=verbose, 
                    coverage=coverage,
                    parallel=parallel
                )
                
                if not success:
                    all_passed = False
                
                progress.update(task, completed=1)
        
        self.end_time = time.time()
        
        # Generate report
        self._generate_report()
        
        return all_passed
    
    def _run_test_type(self, 
                      test_type: str, 
                      verbose: bool = False,
                      coverage: bool = False,
                      parallel: bool = False) -> bool:
        """Run specific test type."""
        
        test_commands = {
            'unit': self._get_unit_test_command,
            'integration': self._get_integration_test_command,
            'performance': self._get_performance_test_command,
            'config': self._get_config_test_command,
            'end_to_end': self._get_e2e_test_command
        }
        
        if test_type not in test_commands:
            console.print(f"[red]Unknown test type: {test_type}[/red]")
            return False
        
        cmd = test_commands[test_type](verbose, coverage, parallel)
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            self.test_results[test_type] = {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
            
            if result.returncode != 0:
                console.print(f"[red]❌ {test_type.title()} tests failed[/red]")
                if verbose:
                    console.print(f"[dim]STDOUT:[/dim]\n{result.stdout}")
                    console.print(f"[dim]STDERR:[/dim]\n{result.stderr}")
                return False
            else:
                console.print(f"[green]✅ {test_type.title()} tests passed[/green]")
                return True
                
        except subprocess.TimeoutExpired:
            console.print(f"[red]❌ {test_type.title()} tests timed out[/red]")
            self.test_results[test_type] = {
                'success': False,
                'error': 'Timeout after 5 minutes'
            }
            return False
        except Exception as e:
            console.print(f"[red]❌ Error running {test_type} tests: {e}[/red]")
            self.test_results[test_type] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def _get_unit_test_command(self, verbose: bool, coverage: bool, parallel: bool) -> List[str]:
        """Get command for unit tests."""
        cmd = ['python', '-m', 'pytest', 'tests/test_enhanced_pipeline.py']
        
        if verbose:
            cmd.append('-v')
        if coverage:
            cmd.extend(['--cov=core', '--cov-report=html', '--cov-report=term'])
        if parallel:
            cmd.extend(['-n', 'auto'])
        
        cmd.extend(['--tb=short', '--strict-markers'])
        return cmd
    
    def _get_integration_test_command(self, verbose: bool, coverage: bool, parallel: bool) -> List[str]:
        """Get command for integration tests."""
        cmd = ['python', '-m', 'pytest', 'tests/test_integration.py']
        
        if verbose:
            cmd.append('-v')
        if coverage:
            cmd.extend(['--cov=core', '--cov-append'])
        
        cmd.extend(['--tb=short', '--asyncio-mode=auto'])
        return cmd
    
    def _get_performance_test_command(self, verbose: bool, coverage: bool, parallel: bool) -> List[str]:
        """Get command for performance tests."""
        cmd = ['python', '-m', 'pytest', 'tests/test_integration.py::TestPerformanceIntegration']
        
        if verbose:
            cmd.append('-v')
        
        cmd.extend(['--tb=short', '--benchmark-only'])
        return cmd
    
    def _get_config_test_command(self, verbose: bool, coverage: bool, parallel: bool) -> List[str]:
        """Get command for configuration tests."""
        cmd = ['python', '-m', 'pytest', 'tests/test_integration.py::TestConfigurationIntegration']
        
        if verbose:
            cmd.append('-v')
        
        cmd.extend(['--tb=short'])
        return cmd
    
    def _get_e2e_test_command(self, verbose: bool, coverage: bool, parallel: bool) -> List[str]:
        """Get command for end-to-end tests."""
        cmd = ['python', '-m', 'pytest', 'tests/test_integration.py::TestEndToEndIntegration']
        
        if verbose:
            cmd.append('-v')
        
        cmd.extend(['--tb=short', '--asyncio-mode=auto'])
        return cmd
    
    def _generate_report(self):
        """Generate comprehensive test report."""
        total_time = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        # Summary table
        table = Table(title="Test Results Summary")
        table.add_column("Test Type", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Details")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get('success', False))
        
        for test_type, result in self.test_results.items():
            status = "[green]✅ PASSED[/green]" if result.get('success', False) else "[red]❌ FAILED[/red]"
            
            details = ""
            if not result.get('success', False):
                if 'error' in result:
                    details = f"Error: {result['error']}"
                elif 'stderr' in result and result['stderr']:
                    details = f"Error output available"
            
            table.add_row(test_type.title(), status, details)
        
        console.print("\n")
        console.print(table)
        
        # Overall summary
        if passed_tests == total_tests:
            summary_color = "green"
            summary_icon = "🎉"
            summary_text = "ALL TESTS PASSED"
        else:
            summary_color = "red"
            summary_icon = "❌"
            summary_text = f"{passed_tests}/{total_tests} TESTS PASSED"
        
        console.print(f"\n[{summary_color}]{summary_icon} {summary_text}[/{summary_color}]")
        console.print(f"[dim]Total execution time: {total_time:.2f} seconds[/dim]")
        
        # Save detailed report
        self._save_detailed_report()
    
    def _save_detailed_report(self):
        """Save detailed test report to file."""
        report_data = {
            'timestamp': time.time(),
            'total_time': self.end_time - self.start_time if self.end_time and self.start_time else 0,
            'results': self.test_results,
            'summary': {
                'total_tests': len(self.test_results),
                'passed_tests': sum(1 for result in self.test_results.values() if result.get('success', False)),
                'failed_tests': sum(1 for result in self.test_results.values() if not result.get('success', False))
            }
        }
        
        report_file = self.project_root / 'test_report.json'
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        console.print(f"\n[dim]Detailed report saved to: {report_file}[/dim]")


def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'pytest', 'pytest-asyncio', 'pytest-cov', 'rich'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        console.print(f"[red]Missing required packages: {', '.join(missing_packages)}[/red]")
        console.print("[yellow]Install with: pip install " + ' '.join(missing_packages) + "[/yellow]")
        return False
    
    return True


def setup_test_environment():
    """Setup test environment and directories."""
    project_root = Path(__file__).parent
    
    # Create test directories if they don't exist
    test_dirs = ['tests', 'input', 'output', 'cache', 'temp', 'logs']
    for dir_name in test_dirs:
        (project_root / dir_name).mkdir(exist_ok=True)
    
    # Create sample test files if they don't exist
    sample_input = project_root / 'input' / 'sample_test.txt'
    if not sample_input.exists():
        sample_input.write_text(
            "This is a sample document for testing the LangGraph document processing pipeline. "
            "It contains multiple sentences and demonstrates the system's ability to process text. "
            "The document discusses artificial intelligence, machine learning, and natural language processing."
        )
    
    console.print("[green]✅ Test environment setup complete[/green]")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='Run LangGraph document processing tests')
    parser.add_argument('--types', nargs='+', 
                       choices=['unit', 'integration', 'performance', 'config', 'end_to_end'],
                       help='Test types to run (default: all)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--coverage', '-c', action='store_true',
                       help='Generate coverage report')
    parser.add_argument('--parallel', '-p', action='store_true',
                       help='Run tests in parallel')
    parser.add_argument('--setup-only', action='store_true',
                       help='Only setup test environment')
    parser.add_argument('--check-deps', action='store_true',
                       help='Only check dependencies')
    
    args = parser.parse_args()
    
    # Check dependencies
    if args.check_deps or not check_dependencies():
        return 1 if not check_dependencies() else 0
    
    # Setup test environment
    setup_test_environment()
    
    if args.setup_only:
        return 0
    
    # Run tests
    project_root = Path(__file__).parent
    runner = TestRunner(project_root)
    
    success = runner.run_all_tests(
        test_types=args.types,
        verbose=args.verbose,
        coverage=args.coverage,
        parallel=args.parallel
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())