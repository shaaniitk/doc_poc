from modules.section_mapper import assign_chunks_to_skeleton


def test_assign_chunks_to_skeleton_maps_introduction():
    grouped = {
        "Introduction": [
            {"content": "Intro content A."},
            {"content": "Intro content B."},
        ]
    }
    assignments = assign_chunks_to_skeleton(grouped)
    assert "1. Introduction" in assignments
    contents = assignments["1. Introduction"][0]["content"]
    assert "Intro content A." in contents and "Intro content B." in contents