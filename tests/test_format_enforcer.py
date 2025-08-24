from modules.format_enforcer import FormatEnforcer

def test_validate_output_detects_markdown_in_latex():
    fe = FormatEnforcer(output_format="latex")
    issues = fe.validate_output("This is **bold** text")
    assert any("textbf" in msg for msg in issues)


def test_post_process_converts_markdown_to_latex():
    fe = FormatEnforcer(output_format="latex")
    fixed = fe.post_process_output("This is **bold** and *italic*\n- item")
    assert "\\textbf{" in fixed
    assert "\\textit{" in fixed
    assert "\\item" in fixed