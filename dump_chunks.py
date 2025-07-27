from modules.chunker import extract_latex_sections

chunks = extract_latex_sections(open('bitcoin_whitepaper.tex').read())

with open('bitcoin_chunks_full.md', 'w', encoding='utf-8') as f:
    f.write('# Bitcoin Whitepaper - Full Chunk Analysis\n\n')
    f.write(f'Total chunks: {len(chunks)}\n\n')
    
    for i, chunk in enumerate(chunks):
        f.write(f'## Chunk {i}\n\n')
        f.write(f'**Type:** {chunk["type"]}\n\n')
        f.write(f'**Parent Section:** {chunk["parent_section"]}\n\n')
        f.write(f'**Content:**\n\n')
        f.write('```\n')
        f.write(chunk['content'])
        f.write('\n```\n\n')
        f.write('---\n\n')

print('Full chunks dumped to bitcoin_chunks_full.md')