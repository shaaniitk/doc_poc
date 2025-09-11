from modules.output_formatter import HierarchicalOutputFormatter as OutputFormatter

of = OutputFormatter('json')
doc = of.format_document({'Intro': 'Hello'})
print('Length:', len(doc))
print('Contains Intro:', '"Intro"' in doc)
print('Test result:', 'PASS' if '"Intro"' in doc else 'FAIL')