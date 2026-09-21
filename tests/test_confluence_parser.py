"""Tests for ingestion/confluence/parser.py — Confluence Storage Format.

The macro-metadata cases are the important ones: the bug this parser exists
to prevent is `ac:parameter` values and `ri:*` attributes being indexed as if
an author had written them.
"""
from ingestion.confluence.parser import ConfluenceHtmlParser

parser = ConfluenceHtmlParser()


def test_headings_and_paragraphs_become_separate_blocks():
    text = parser.parse("<h1>Глоссарий</h1><p>Первый абзац.</p><p>Второй абзац.</p>")
    assert text == "Глоссарий\n\nПервый абзац.\n\nВторой абзац."


def test_all_heading_levels_are_kept():
    html = "".join(f"<h{i}>Heading {i}</h{i}>" for i in range(1, 7))
    text = parser.parse(html)
    for i in range(1, 7):
        assert f"Heading {i}" in text


def test_lists_become_dash_items():
    text = parser.parse("<ul><li>Резервуар</li><li>Датчик</li></ul><ol><li>Первый</li></ol>")
    assert "- Резервуар" in text
    assert "- Датчик" in text
    assert "- Первый" in text


def test_table_rows_become_pipe_separated_lines():
    html = """
    <table><tbody>
      <tr><th>Привилегия</th><th>Описание</th></tr>
      <tr><td>admin</td><td>Полный доступ</td></tr>
    </tbody></table>
    """
    text = parser.parse(html)
    assert "Привилегия | Описание" in text
    assert "admin | Полный доступ" in text


def test_code_blocks_keep_their_lines():
    html = "<pre>line one\nline two</pre><p>after</p>"
    text = parser.parse(html)
    assert "line one" in text
    assert "line two" in text


def test_div_span_br_are_traversed_not_dropped():
    text = parser.parse("<div><span>alpha</span><br/><span>beta</span></div>")
    assert "alpha" in text
    assert "beta" in text


def test_rich_text_body_content_is_kept():
    """ac:rich-text-body holds real authored text inside a macro (info panels,
    expand blocks) — dropping it would lose page content."""
    html = """
    <ac:structured-macro ac:name="info">
      <ac:rich-text-body><p>Важное примечание для оператора.</p></ac:rich-text-body>
    </ac:structured-macro>
    """
    assert "Важное примечание для оператора." in parser.parse(html)


def test_plain_text_body_content_is_kept_as_code():
    html = """
    <ac:structured-macro ac:name="code">
      <ac:plain-text-body>GET /rest/api/content</ac:plain-text-body>
    </ac:structured-macro>
    """
    assert "GET /rest/api/content" in parser.parse(html)


def test_ac_parameter_metadata_never_reaches_the_text():
    """Regression for the reference project's bug: macro/blueprint metadata
    being indexed as user text."""
    html = """
    <ac:structured-macro ac:name="info">
      <ac:parameter ac:name="title">MACRO-TITLE-METADATA</ac:parameter>
      <ac:parameter ac:name="layout">two-column-right-sidebar</ac:parameter>
      <ac:rich-text-body><p>Настоящий текст страницы.</p></ac:rich-text-body>
    </ac:structured-macro>
    """
    text = parser.parse(html)
    assert "Настоящий текст страницы." in text
    assert "MACRO-TITLE-METADATA" not in text
    assert "two-column-right-sidebar" not in text


def test_ri_elements_are_dropped():
    html = """
    <p>До ссылки.</p>
    <ac:link><ri:page ri:content-title="RI-PAGE-METADATA"/></ac:link>
    <ac:image><ri:attachment ri:filename="RI-ATTACHMENT-METADATA.png"/></ac:image>
    <p>После ссылки.</p>
    """
    text = parser.parse(html)
    assert "До ссылки." in text
    assert "После ссылки." in text
    assert "RI-PAGE-METADATA" not in text
    assert "RI-ATTACHMENT-METADATA" not in text


def test_script_style_and_navigation_are_dropped():
    html = """
    <script>var leak = 'SCRIPT-BODY';</script>
    <style>.x { color: red }</style>
    <nav>NAVIGATION-CHROME</nav>
    <div role="navigation">ROLE-NAVIGATION</div>
    <p>Содержимое.</p>
    """
    text = parser.parse(html)
    assert text == "Содержимое."


def test_layout_macros_are_descended_into():
    html = """
    <ac:layout><ac:layout-section ac:type="two_equal">
      <ac:layout-cell><p>Левая колонка.</p></ac:layout-cell>
      <ac:layout-cell><p>Правая колонка.</p></ac:layout-cell>
    </ac:layout-section></ac:layout>
    """
    text = parser.parse(html)
    assert "Левая колонка." in text
    assert "Правая колонка." in text


def test_empty_html_yields_empty_string():
    assert parser.parse("") == ""
