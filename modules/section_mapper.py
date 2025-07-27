"""Section mapping and assignment module"""

from config import DOCUMENT_TEMPLATES

def get_document_skeleton(template_name="bitcoin_paper"):
    return DOCUMENT_TEMPLATES.get(template_name, DOCUMENT_TEMPLATES["bitcoin_paper"])

DOCUMENT_SKELETON = get_document_skeleton()

def assign_chunks_to_skeleton(grouped_chunks):
    """Assign grouped chunks to skeleton sections"""
    assignments = {s['section']: [] for s in DOCUMENT_SKELETON}
    
    section_mapping = {
        'Abstract': 'Abstract',
        'Introduction': '1. Introduction', 
        'Transactions': '2. Transactions',
        'Timestamp Server': '3. Timestamp Server',
        'Proof-of-Work': '4. Proof-of-Work',
        'Network': '5. Network',
        'Incentive': '6. Incentive',
        'Reclaiming Disk Space': '7. Reclaiming Disk Space',
        'Simplified Payment Verification': '8. Simplified Payment Verification',
        'Combining and Splitting Value': '9. Combining and Splitting Value',
        'Privacy': '10. Privacy',
        'Calculations': '12. Calculations',
        'Conclusion': '13. Conclusion'
    }
    
    # Special handling for Summary section - gets all content
    if 'Summary' in assignments:
        all_content = []
        for section_chunks in grouped_chunks.values():
            for chunk in section_chunks:
                all_content.append(chunk['content'])
        assignments['Summary'].append({
            'content': '\n\n'.join(all_content),
            'parent_section': 'All Sections'
        })
    
    # Special handling for Major and Minor Assumptions - gets all content
    if '11. Major and Minor Assumptions' in assignments:
        all_content = []
        for section_chunks in grouped_chunks.values():
            for chunk in section_chunks:
                all_content.append(chunk['content'])
        assignments['11. Major and Minor Assumptions'].append({
            'content': '\n\n'.join(all_content),
            'parent_section': 'All Sections'
        })
    
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