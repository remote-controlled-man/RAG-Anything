"""Opt-in limits for model input bodies, preserving complete table rows."""

import re
from html.parser import HTMLParser


class TokenBudgetError(ValueError):
    """The requested budget cannot contain the mandatory structure and marker."""


class _HeaderRowParser(HTMLParser):
    """Identify column-only header rows when no explicit thead is present."""

    def __init__(self):
        super().__init__()
        self.has_column_header = False
        self.has_data_cell = False

    def handle_starttag(self, tag, attrs):
        if tag == "td":
            self.has_data_cell = True
        elif tag == "th":
            scope = (dict(attrs).get("scope") or "").strip().lower()
            if scope in {"row", "rowgroup"}:
                self.has_data_cell = True
            else:
                self.has_column_header = True


class _TableHeadParser(HTMLParser):
    """Count explicit header rows without treating comments or raw text as tags."""

    def __init__(self):
        super().__init__()
        self.header_count = None
        self.in_thead = False

    def handle_starttag(self, tag, attrs):
        if tag == "thead" and self.header_count is None:
            self.header_count = 0
            self.in_thead = True
        elif tag == "tr" and self.in_thead:
            self.header_count += 1

    def handle_endtag(self, tag):
        if tag == "thead":
            self.in_thead = False


def truncate_text(text, budget, count_tokens, marker):
    if budget <= 0 or count_tokens(text) <= budget:
        return text
    if count_tokens(marker) > budget:
        raise TokenBudgetError(
            "Token budget is too small for the truncation marker; increase it or use 0"
        )
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if count_tokens(text[:mid] + marker) <= budget:
            low = mid
        else:
            high = mid - 1
    return text[:low] + marker


def _fit_rows(rows, render, budget, count_tokens):
    # Formatting can remove overhead without dropping any rows.
    complete = render(len(rows))
    if count_tokens(complete) <= budget:
        return complete
    if count_tokens(render(0)) > budget:
        raise TokenBudgetError(
            "Token budget cannot fit the table header and omission marker; increase it or use 0"
        )
    low, high = 0, len(rows) - 1
    while low < high:
        mid = (low + high + 1) // 2
        if count_tokens(render(mid)) <= budget:
            low = mid
        else:
            high = mid - 1
    return render(low)


def truncate_table(body, budget, count_tokens):
    if budget <= 0 or count_tokens(body) <= budget:
        return body
    if "<table" in body.lower():
        if len(re.findall(r"<table\b", body, re.IGNORECASE)) != 1:
            raise TokenBudgetError(
                "Split nested or multiple tables before applying a token budget"
            )
        row_pattern = r"<tr\b[^>]*>.*?</tr\s*>"
        rows = re.findall(row_pattern, body, re.IGNORECASE | re.DOTALL)
        if not rows:
            raise TokenBudgetError("HTML table has no complete rows to retain")
        head = _TableHeadParser()
        head.feed(body)
        head.close()
        if head.header_count is not None:
            header_count = head.header_count
        else:
            header_count = 0
            for row in rows:
                cells = _HeaderRowParser()
                cells.feed(row)
                if not cells.has_column_header or cells.has_data_cell:
                    break
                header_count += 1
        header_count = max(1, header_count)
        header, data = "".join(rows[:header_count]), rows[header_count:]
        caption = re.search(
            r"<caption\b[^>]*>.*?</caption\s*>", body, re.IGNORECASE | re.DOTALL
        )

        def render(n):
            omitted = len(data) - n
            marker = (
                f"<!-- {omitted} rows omitted for context budget; total rows: {len(data)} -->"
                if omitted
                else ""
            )
            return (
                "<table>"
                + (caption.group() if caption else "")
                + "<tbody>"
                + header
                + "".join(data[:n])
                + marker
                + "</tbody></table>"
            )

        return _fit_rows(data, render, budget, count_tokens)

    lines = body.splitlines()
    if len(lines) >= 2 and "|" in lines[0] and re.fullmatch(r"[\s|:\-]+", lines[1]):
        header, data = "\n".join(lines[:2]), lines[2:]

        def render(n):
            omitted = len(data) - n
            marker = (
                f"\n[... {omitted} rows omitted for LLM context window budget; total rows: {len(data)} ...]"
                if omitted
                else ""
            )
            return header + "\n" + "\n".join(data[:n]) + marker

        return _fit_rows(data, render, budget, count_tokens)

    if len(lines) > 1:
        header, data = lines[0], lines[1:]

        def render(n):
            omitted = len(data) - n
            marker = (
                f"\n[... {omitted} lines omitted for context budget ...]"
                if omitted
                else ""
            )
            return header + "\n" + "\n".join(data[:n]) + marker

        return _fit_rows(data, render, budget, count_tokens)
    return truncate_text(body, budget, count_tokens, "\n[... content truncated ...]")
