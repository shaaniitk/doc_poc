#!/usr/bin/env python3
"""
Fix the conftest.py stubbed embedding model to properly assign both test chunks to Introduction.
The issue is that 'text2' doesn't contain introduction keywords, so it gets a generic embedding
and the section mapper assigns it to 'Conclusion' instead of '1. Introduction'.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'tests')))

# Load conftest to activate stubbing
import conftest

# Test the current behavior
from modules.section_mapper import assign_chunks_to_skeleton, SemanticMapper

# Test data from the failing test (exact format)
grouped = {
    "Introduction": [
        {"content": "Commerce on the Internet has come to rely almost exclusively on financial institutions serving as trusted third parties to process electronic payments. The traditional trust-based model has inherent weaknesses."},
        {"content": "What is needed is an electronic payment system based on cryptographic proof instead of trust, allowing any two willing parties to transact directly with each other without the need for a trusted third party."},
    ]
}

print("=== Current Behavior (Exact Test Format) ===")
result = assign_chunks_to_skeleton(grouped)

for section, assignments in result.items():
    print(f"Section: {section}")
    if assignments:
        content = assignments[0]["content"]
        print(f"Content length: {len(content)}")
        print(f"Content preview: {content[:100]}...")
        print(f"Contains first text: {'Commerce on the Internet' in content}")
        print(f"Contains second text: {'What is needed is an electronic payment system' in content}")
    else:
        print("No assignments")
    print()

print("=== Analysis ===")
print("The issue is that the second chunk doesn't contain enough introduction keywords,")
print("so the stubbed model gives it a generic embedding that gets assigned to 'Conclusion'.")
print("We need to either:")
print("1. Update the stubbed model to recognize both chunks as introduction content")
print("2. Modify the stubbed model to be less strict about keyword matching")
print("3. Fix the section mapper to handle generic embeddings better")

# Test the exact assertion from the failing test
if "1. Introduction" in result:
    contents = result["1. Introduction"][0]["content"]
    first_text_present = "Commerce on the Internet has come to rely almost exclusively on financial institutions serving as trusted third parties to process electronic payments. The traditional trust-based model has inherent weaknesses." in contents
    second_text_present = "What is needed is an electronic payment system based on cryptographic proof instead of trust, allowing any two willing parties to transact directly with each other without the need for a trusted third party." in contents
    
    print(f"\n=== Test Assertion Check ===")
    print(f"First text present: {first_text_present}")
    print(f"Second text present: {second_text_present}")
    print(f"Test would {'PASS' if first_text_present and second_text_present else 'FAIL'}")
else:
    print("\n=== Test Assertion Check ===")
    print("'1. Introduction' section not found - Test would FAIL")