#!/usr/bin/env python3
"""Helper script to run Mistral integration tests with cost control."""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🚀 Mistral API Integration Test Runner")
    print("=" * 50)
    
    # Check if API key is set
    api_key = os.getenv('MISTRAL_API_KEY')
    
    if not api_key:
        print("\n❌ MISTRAL_API_KEY environment variable not found!")
        print("\n📝 To set it:")
        print("   Windows PowerShell: $env:MISTRAL_API_KEY='your_api_key_here'")
        print("   Windows CMD: set MISTRAL_API_KEY=your_api_key_here")
        print("   Linux/Mac: export MISTRAL_API_KEY=your_api_key_here")
        
        # Interactive input option
        user_input = input("\n🔑 Enter your Mistral API key now (or press Enter to skip): ").strip()
        if user_input:
            os.environ['MISTRAL_API_KEY'] = user_input
            api_key = user_input
            print("✅ API key set for this session")
        else:
            print("⚠️  Skipping test - no API key provided")
            return
    
    # Show cost estimate
    print(f"\n💰 Cost Estimate:")
    print(f"   Model: mistral-small (cheapest option)")
    print(f"   Max tokens per call: 50 (very limited)")
    print(f"   Max API calls: 2 (minimal testing)")
    print(f"   Estimated cost: < $0.005 USD (less than half a cent)")
    
    # Show what will be tested
    print(f"\n🧪 Test Plan:")
    print(f"   1. Single API call test (basic connectivity)")
    print(f"   2. Template processing test (integration)")
    print(f"   3. Automatic stop after 2 calls")
    
    # Confirm before proceeding
    confirm = input("\n🤔 Proceed with minimal cost test? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("❌ Test cancelled by user")
        return
    
    # Check if test file exists
    test_file = Path(__file__).parent / "test_mistral_integration.py"
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return
    
    # Check if required dependencies exist
    required_files = [
        "mistral_llm_handler.py",
        "core/template_processor_node.py",
        "core/models.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not (Path(__file__).parent / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n💡 Make sure all core components are available")
    
    print("\n🔄 Running Mistral integration test...")
    print("-" * 30)
    
    try:
        # Run the test
        result = subprocess.run(
            [sys.executable, str(test_file)],
            cwd=str(test_file.parent),
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        if result.returncode == 0:
            print("\n✅ Test completed successfully!")
            print("\n🎉 Your Mistral API integration is working!")
        else:
            print(f"\n❌ Test failed with exit code: {result.returncode}")
            print("\n💡 Check the error messages above for troubleshooting")
            
    except Exception as e:
        print(f"\n❌ Error running test: {e}")
    
    print("\n📊 Test session complete")
    print("💰 Estimated cost incurred: < $0.005 USD")

if __name__ == "__main__":
    main()