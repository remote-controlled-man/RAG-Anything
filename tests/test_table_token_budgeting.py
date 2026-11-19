"""Unit tests for token budgeting and structure-preserving table truncation (Issue #83).

Verifies that:
1. Tables and equations exceeding token budgets are truncated defensively.
2. Markdown table structure (header row and separator) is preserved while retaining
   data rows up to the configured token budget.
3. HTML tables are truncated at row boundaries and closed cleanly.
4. An explicit structural summary marker is appended documenting the omitted rows.
5. End-to-end captioning functions receive bounded prompts preventing LLM context overflows.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from raganything.config import RAGAnythingConfig
from raganything.modalprocessors import EquationModalProcessor, TableModalProcessor


def create_mock_lightrag(tokenizer=None):
    lr = MagicMock()
    lr.text_chunks = MagicMock()
    lr.chunks_vdb = MagicMock()
    lr.entities_vdb = MagicMock()
    lr.relationships_vdb = MagicMock()
    lr.chunk_entity_relation_graph = MagicMock()
    lr.embedding_func = MagicMock()
    lr.llm_model_func = MagicMock()
    lr._build_global_config = MagicMock(return_value={})
    lr.llm_response_cache = MagicMock()
    lr.tokenizer = tokenizer
    return lr


def test_config_token_budget_defaults():
    config = RAGAnythingConfig()
    assert config.max_table_tokens == 0
    assert config.max_modal_content_tokens == 0


@pytest.mark.parametrize("kind", ["markdown", "html", "equation"])
def test_explicit_budget_counts_the_marker_and_multibyte_text(kind):
    from raganything.token_budget import truncate_table, truncate_text

    def count(text):
        return len(text.encode("utf-8"))

    budget = 300
    if kind == "equation":
        result = truncate_text(
            "变量甲乙丙" * 200, budget, count, "\n[... truncated ...]"
        )
    else:
        content = (
            ("| 项目 | 数量 |\n|---|---|\n" + "| 中文项目 | 42 |\n" * 100)
            if kind == "markdown"
            else (
                "<table><tr><th>项目</th><th>数量</th></tr>"
                + "<tr><td>中文项目</td><td>42</td></tr>" * 100
                + "</table>"
            )
        )
        result = truncate_table(content, budget, count)
    assert count(result) <= budget
    assert "�" not in result


@pytest.mark.asyncio
async def test_budget_too_small_is_an_error_and_never_a_successful_caption():
    from raganything.token_budget import TokenBudgetError

    model = AsyncMock()
    processor = TableModalProcessor(create_mock_lightrag(), model, max_table_tokens=1)
    with pytest.raises(TokenBudgetError, match="header"):
        await processor.generate_description_only(
            {"table_body": "| Long header |\n|---|\n" + "| a value |\n" * 20}, "table"
        )
    model.assert_not_called()


def test_table_small_markdown_passes_through():
    lr = create_mock_lightrag()
    proc = TableModalProcessor(
        lightrag=lr, modal_caption_func=AsyncMock(), max_table_tokens=100
    )
    small_md = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
    truncated = proc._truncate_table_for_context(small_md)
    assert truncated == small_md


def test_table_large_markdown_structure_preserved():
    lr = create_mock_lightrag()
    # Token budget of 50 tokens (approx 200 chars)
    proc = TableModalProcessor(
        lightrag=lr, modal_caption_func=AsyncMock(), max_table_tokens=50
    )

    rows = [
        f"| Item {i} | Category {i} | Description for value {i} in table |"
        for i in range(100)
    ]
    large_md = "| ID | Name | Details |\n|---|---|---|\n" + "\n".join(rows)

    truncated = proc._truncate_table_for_context(large_md)

    # Header and separator MUST be preserved
    assert "| ID | Name | Details |" in truncated
    assert "|---|---|---|" in truncated

    # Truncation marker must be present with row counts
    assert "rows omitted for LLM context window budget" in truncated
    assert "total rows: 100" in truncated

    # Total token count must be within reasonable bounds
    assert proc._count_tokens(truncated) <= 65


def test_table_large_html_structure_preserved():
    lr = create_mock_lightrag()
    proc = TableModalProcessor(
        lightrag=lr, modal_caption_func=AsyncMock(), max_table_tokens=60
    )

    html_rows = [
        f"<tr><td>{i}</td><td>Value_{i}</td><td>Extra long text details for row {i}</td></tr>"
        for i in range(80)
    ]
    large_html = (
        "<table><thead><tr><th>ID</th><th>Val</th><th>Desc</th></tr></thead><tbody>\n"
        + "\n".join(html_rows)
        + "</tbody></table>"
    )

    truncated = proc._truncate_table_for_context(large_html)

    # Header must be present
    assert "<th>ID</th>" in truncated
    # Closing tags and comments must be present
    assert "rows omitted for context budget" in truncated
    assert "</tbody></table>" in truncated


def test_table_disable_truncation():
    lr = create_mock_lightrag()
    proc = TableModalProcessor(
        lightrag=lr, modal_caption_func=AsyncMock(), max_table_tokens=0
    )
    large_text = "| Col |\n|---|\n" + "\n".join([f"| row {i} |" for i in range(200)])
    # When max_table_tokens is 0, no truncation occurs
    assert proc._truncate_table_for_context(large_text) == large_text


def test_equation_truncation():
    lr = create_mock_lightrag()
    proc = EquationModalProcessor(
        lightrag=lr, modal_caption_func=AsyncMock(), max_modal_content_tokens=30
    )

    long_eq = "x = " + " + ".join([f"a_{i} * y_{i}" for i in range(100)])
    truncated = proc._truncate_equation_for_context(long_eq)

    assert "equation truncated to ~30 tokens" in truncated
    assert len(truncated) < len(long_eq)


@pytest.mark.asyncio
async def test_table_generate_description_only_respects_token_budget():
    lr = create_mock_lightrag()
    caption_calls = []

    async def mock_caption(prompt, system_prompt=None):
        caption_calls.append(prompt)
        return '{"detailed_description": "Analyzed table", "entity_info": {"entity_name": "T1", "entity_type": "table", "summary": "Sum"}}'

    proc = TableModalProcessor(
        lightrag=lr, modal_caption_func=mock_caption, max_table_tokens=80
    )

    rows = [
        f"| Row {i} | Value {i * 10} | Extremely descriptive narrative for line {i} |"
        for i in range(150)
    ]
    massive_table = "| Col A | Col B | Col C |\n|---|---|---|\n" + "\n".join(rows)

    desc, entity_info = await proc.generate_description_only(
        modal_content={"table_body": massive_table},
        content_type="table",
    )

    assert len(caption_calls) == 1
    prompt_sent = caption_calls[0]

    # Verify the table in the prompt was truncated and includes the summary marker
    assert "rows omitted for LLM context window budget" in prompt_sent
    assert desc == "Analyzed table"
    assert entity_info["entity_name"] == "T1 (table)"


@pytest.mark.asyncio
async def test_html_row_headers_are_truncated_before_captioning():
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda text: list(text.encode("utf-8"))
    caption = AsyncMock(
        return_value='{"detailed_description": "Regional sales", "entity_info": '
        '{"entity_name": "Sales", "entity_type": "table", "summary": "Sales data"}}'
    )
    processor = TableModalProcessor(
        create_mock_lightrag(tokenizer), caption, max_table_tokens=350
    )
    header = "<tr><th>Region</th><th>Sales</th></tr>"
    rows = [f'<tr><th scope="row">Region {i}</th><td>{i}</td></tr>' for i in range(20)]
    table = (
        f"<table><thead>{header}</thead><tbody>" + "".join(rows) + "</tbody></table>"
    )

    description, _ = await processor.generate_description_only(
        {"table_body": table}, "table"
    )

    caption.assert_awaited_once()
    prompt = caption.await_args.args[0]
    sent_table = prompt[prompt.index("<table>") : prompt.index("</table>") + 8]
    assert len(sent_table.encode("utf-8")) <= 350
    assert header in sent_table
    assert rows[0] in sent_table
    assert rows[-1] not in sent_table
    assert "total rows: 20" in sent_table
    assert description == "Regional sales"


@pytest.mark.parametrize(
    "cells",
    [
        "<th>Region {i}</th><td>{i}</td>",
        '<th scope="row">Region {i}</th><th scope="row">{i}</th>',
        "<th scope='ROWGROUP'>Region {i}</th><th scope='rowgroup'>{i}</th>",
        "<th scope=row>Region {i}</th><th scope=row>{i}</th>",
    ],
    ids=["mixed-cells", "row-scope", "rowgroup-scope", "unquoted-scope"],
)
def test_html_inferred_headers_allow_row_header_truncation(cells):
    from raganything.token_budget import truncate_table

    header = (
        '<tr><th colspan="2" scope="colgroup">Sales</th></tr>'
        "<tr><th>Region</th><th>Revenue</th></tr>"
    )
    rows = [f"<tr>{cells.format(i=i)}</tr>" for i in range(20)]
    table = f"<table>{header}" + "".join(rows) + "</table>"

    result = truncate_table(table, 400, len)

    assert len(result) <= 400
    assert header in result
    assert rows[0] in result
    assert rows[-1] not in result
    assert "total rows: 20" in result
    assert result.count("<tr>") == result.count("</tr>")
    assert result.endswith("</tbody></table>")


def test_html_explicit_header_section_preserves_mixed_cells_and_caption():
    from raganything.token_budget import truncate_table

    caption = "<caption>Annual sales</caption>"
    header = (
        '<tr><td></td><th colspan="2">Revenue</th></tr>'
        "<tr><th>Region</th><th>2025</th><th>2026</th></tr>"
    )
    rows = [
        f'<tr><th scope="row">Region {i}</th><td>{i}</td><td>{i + 1}</td></tr>'
        for i in range(20)
    ]
    table = (
        f'<table>{caption}<thead class="columns">{header}</thead><tbody>'
        + "".join(rows)
        + "</tbody></table>"
    )

    result = truncate_table(table, 400, len)

    assert len(result) <= 400
    assert caption in result
    assert header in result
    assert rows[0] in result
    assert rows[-1] not in result
    assert "total rows: 20" in result


@pytest.mark.parametrize("budget", [0, 10000], ids=["disabled", "fits"])
def test_html_row_header_table_passes_through_when_it_fits(budget):
    from raganything.token_budget import truncate_table

    table = (
        "<table><thead><tr><th>Region</th><th>Revenue</th></tr></thead>"
        '<tbody><tr><th scope="row">North</th><td>42</td></tr></tbody></table>'
    )

    assert truncate_table(table, budget, len) == table


@pytest.mark.parametrize(
    "annotation",
    [
        "<!-- <thead></thead> -->",
        '<script>const example = "<thead></thead>";</script>',
        "<style>/* <thead></thead> */</style>",
    ],
    ids=["comment", "script", "style"],
)
@pytest.mark.parametrize("budget", [130, 185], ids=["header-too-large", "fits"])
def test_html_header_inference_ignores_non_markup_thead_text(annotation, budget):
    from raganything.token_budget import TokenBudgetError, truncate_table

    header = "<tr><th>Sales</th></tr><tr><th>Revenue</th></tr>"
    rows = [f"<tr><td>Value {i}</td></tr>" for i in range(20)]
    table = f"<table>{annotation}{header}" + "".join(rows) + "</table>"

    if budget == 130:
        with pytest.raises(TokenBudgetError, match="header"):
            truncate_table(table, budget, len)
        return

    result = truncate_table(table, budget, len)

    assert len(result) <= budget
    assert header in result
    assert "total rows: 20" in result


@pytest.mark.parametrize(
    "annotation",
    [
        "<!-- <thead><tr><th>Example</th></tr></thead> -->",
        "<!-- <thead><tr> -->",
        '<script>const example = "<thead><tr><th>Example</th></tr></thead>";</script>',
        "<style>/* <thead><tr><th>Example</th></tr></thead> */</style>",
    ],
    ids=["comment", "incomplete-comment-row", "script", "style"],
)
def test_html_header_count_and_rows_use_the_same_markup(annotation):
    from raganything.token_budget import truncate_table

    header = "<tr><th>Sales</th></tr><tr><th>Revenue</th></tr>"
    rows = [f"<tr><td>Value {i}</td></tr>" for i in range(20)]
    table = (
        f"<table>{annotation}<thead>{header}</thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )

    result = truncate_table(table, 300, len)

    assert len(result) <= 300
    assert header in result
    assert rows[0] in result
    assert "total rows: 20" in result


@pytest.mark.parametrize("group", ["tbody", "tfoot"])
def test_html_row_group_start_ends_an_implicit_thead(group):
    from raganything.token_budget import truncate_table

    header = "<tr><th>Sales</th></tr><tr><th>Revenue</th></tr>"
    rows = [f"<tr><td>Value {i}</td></tr>" for i in range(20)]
    table = f"<table><thead>{header}<{group}>" + "".join(rows) + f"</{group}></table>"

    result = truncate_table(table, 300, len)

    assert len(result) <= 300
    assert header in result
    assert rows[0] in result
    assert "total rows: 20" in result


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_html_row_source_offsets_preserve_multiline_unicode_markup(newline):
    from raganything.token_budget import truncate_table

    header = f"<tr>{newline}<th>区域</th><th>Revenue</th>{newline}</tr >"
    rows = [
        f'<tr data-label="a > b">{newline}<th scope="row">北方 {i}</th>'
        f"<td>{i}</td>{newline}</tr >"
        for i in range(20)
    ]
    table = (
        f"<table>{newline}<thead>{header}</thead>{newline}<tbody>"
        + newline.join(rows)
        + "</tbody></table>"
    )

    result = truncate_table(table, 500, lambda text: len(text.encode("utf-8")))

    assert len(result.encode("utf-8")) <= 500
    assert header in result
    assert rows[0] in result
    assert rows[-1] not in result
    assert "total rows: 20" in result
