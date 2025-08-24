from modules.output_formatter import OutputFormatter


def test_format_latex_document_contains_wrapper():
    of = OutputFormatter("latex")
    doc = of.format_document({"Intro": "Hello"})
    assert "\\begin{document}" in doc and "Hello" in doc


def test_format_markdown_document_contains_headers():
    of = OutputFormatter("markdown")
    doc = of.format_document({"Intro": "Hello"})
    assert "# Intro" in doc and "Hello" in doc


def test_format_json_document_is_serializable():
    of = OutputFormatter("json")
    doc = of.format_document({"Intro": "Hello"})
    assert '"Intro"' in doc