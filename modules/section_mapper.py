"""Section mapping and assignment module"""

from config import DOCUMENT_TEMPLATES
DOCUMENT_SKELETON = DOCUMENT_TEMPLATES["academic_paper"]

def assign_chunks_to_skeleton(grouped_chunks):
    """Assign grouped chunks to skeleton sections"""
    assignments = {s['section']: [] for s in DOCUMENT_SKELETON}
    
    section_mapping = {
        'Abstract': 'Abstract',
        'Introduction': '1. Introduction', 
        'Theoretical Foundations': '2. Theoretical Foundations',
        'The Proposed Framework: DAWF': '3. The Proposed Framework: DAWF',
        'Input Data and Database': '4. Input Data and Database',
        'Implementation Details': '5. Implementation Details',
        'Testing and Verification': '6. Testing and Verification',
        'Experimental Results and Discussion': '7. Experimental Results and Discussion',
        'Conclusion': '8. Conclusion',
        'References': 'References'
    }
    
    for source_section, target_section in section_mapping.items():
        if source_section in grouped_chunks and target_section in assignments:
            combined_content = []
            for chunk in grouped_chunks[source_section]:
                combined_content.append(chunk['content'])
            
            assignments[target_section].append({
                'content': '\n\n'.join(combined_content),
                'parent_section': source_section
            })
    
    return assignments

def get_section_prompt(section_name):
    """Get prompt for a specific section"""
    for skeleton_section in DOCUMENT_SKELETON:
        if skeleton_section['section'] == section_name:
            return skeleton_section['prompt']
    return "Process the following content appropriately."