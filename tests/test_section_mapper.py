from modules.section_mapper import assign_chunks_to_skeleton


def test_assign_chunks_to_skeleton_maps_introduction():
    grouped = {
        "Introduction": [
            {"content": "Commerce on the Internet has come to rely almost exclusively on financial institutions serving as trusted third parties to process electronic payments. The traditional trust-based model has inherent weaknesses."},
            {"content": "What is needed is an electronic payment system based on cryptographic proof instead of trust, allowing any two willing parties to transact directly with each other without the need for a trusted third party."},
        ]
    }
    assignments = assign_chunks_to_skeleton(grouped)
    assert "1. Introduction" in assignments
    contents = assignments["1. Introduction"][0]["content"]
    assert "Commerce on the Internet has come to rely almost exclusively on financial institutions serving as trusted third parties to process electronic payments. The traditional trust-based model has inherent weaknesses." in contents and "What is needed is an electronic payment system based on cryptographic proof instead of trust, allowing any two willing parties to transact directly with each other without the need for a trusted third party." in contents