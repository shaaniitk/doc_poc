# Exact mimic of the pytest test
from modules.section_mapper import assign_chunks_to_skeleton

def test_assign_chunks_to_skeleton_maps_introduction():
    grouped = {
        "Introduction": [
            {"content": "Commerce on the Internet has come to rely almost exclusively on financial institutions serving as trusted third parties to process electronic payments. The traditional trust-based model has inherent weaknesses."},
            {"content": "What is needed is an electronic payment system based on cryptographic proof instead of trust, allowing any two willing parties to transact directly with each other without the need for a trusted third party."},
        ]
    }
    assignments = assign_chunks_to_skeleton(grouped)
    
    print(f"Assignments keys: {list(assignments.keys())}")
    print(f"'1. Introduction' in assignments: {'1. Introduction' in assignments}")
    
    if "1. Introduction" in assignments:
        contents = assignments["1. Introduction"][0]["content"]
        print(f"Contents length: {len(contents)}")
        print(f"Contents: '{contents}'")
        
        text1 = "Commerce on the Internet has come to rely almost exclusively on financial institutions serving as trusted third parties to process electronic payments. The traditional trust-based model has inherent weaknesses."
        text2 = "What is needed is an electronic payment system based on cryptographic proof instead of trust, allowing any two willing parties to transact directly with each other without the need for a trusted third party."
        
        print(f"Text1 in contents: {text1 in contents}")
        print(f"Text2 in contents: {text2 in contents}")
        
        assertion_result = text1 in contents and text2 in contents
        print(f"Assertion result: {assertion_result}")
        
        # This is the exact assertion from the test
        assert "1. Introduction" in assignments
        assert text1 in contents and text2 in contents
        print("[PASS] All assertions passed!")
    else:
        print("✗ '1. Introduction' not found")
        assert False, "'1. Introduction' not found in assignments"

if __name__ == "__main__":
    test_assign_chunks_to_skeleton_maps_introduction()