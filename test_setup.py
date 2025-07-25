#!/usr/bin/env python3
"""
Test script to verify the Hugging Face model callers work correctly.
"""

import sys
import os

def test_mcp_caller():
    """Test the MCP-style API caller."""
    print("🧪 Testing MCP API Caller...")
    
    try:
        # Import the MCP caller
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from hf_mcp_caller import HuggingFaceMCPCaller
        
        # Initialize caller
        caller = HuggingFaceMCPCaller()
        
        # Test model listing
        models = caller.list_models()
        print(f"✅ Models available: {list(models.keys())}")
        
        # Test a simple generation (this might take a moment)
        print("🔮 Testing text generation with Mistral 7B...")
        response = caller.generate_text(
            "mistral-7b",
            "Hello, how are you?",
            max_new_tokens=50
        )
        
        if response:
            print(f"✅ Generation successful: {response[:100]}...")
            return True
        else:
            print("❌ Generation failed")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_config():
    """Test configuration loading."""
    print("\n🧪 Testing Configuration...")
    
    try:
        from config import MODELS, API_CONFIG, DEFAULT_PARAMS
        
        print(f"✅ Found {len(MODELS)} models in config")
        print(f"✅ API config loaded: {API_CONFIG['base_url']}")
        print(f"✅ Default params loaded: {DEFAULT_PARAMS}")
        return True
        
    except ImportError as e:
        print(f"❌ Config import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_requirements():
    """Test if required packages are available."""
    print("\n🧪 Testing Requirements...")
    
    required_packages = [
        ("requests", "HTTP requests"),
        ("json", "JSON handling"),
        ("os", "Operating system interface"),
        ("logging", "Logging functionality")
    ]
    
    missing_packages = []
    
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - {description}")
        except ImportError:
            print(f"❌ {package} - {description}")
            missing_packages.append(package)
    
    # Test optional packages
    optional_packages = [
        ("torch", "PyTorch for local models"),
        ("transformers", "Hugging Face transformers")
    ]
    
    for package, description in optional_packages:
        try:
            __import__(package)
            print(f"✅ {package} - {description}")
        except ImportError:
            print(f"⚠️  {package} - {description} (optional)")
    
    return len(missing_packages) == 0

def main():
    """Run all tests."""
    print("🚀 Running Hugging Face Model Caller Tests")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test requirements
    if test_requirements():
        tests_passed += 1
    
    # Test configuration
    if test_config():
        tests_passed += 1
    
    # Test MCP caller (this requires internet)
    print("\n⚠️  The next test requires internet connection and may take a moment...")
    user_input = input("Do you want to test the API caller? (y/n): ").lower().strip()
    
    if user_input == 'y':
        if test_mcp_caller():
            tests_passed += 1
    else:
        print("⏭️  Skipping API test")
        total_tests -= 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your setup is ready to use.")
        print("\n💡 Next steps:")
        print("   python3 hf_mcp_caller.py          # Run demo")
        print("   python3 hf_mcp_caller.py --chat   # Interactive chat")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("\n🔧 Try running the setup script:")
        print("   ./setup.sh")

if __name__ == "__main__":
    main()
