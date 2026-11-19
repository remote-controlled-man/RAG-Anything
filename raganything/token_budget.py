"""Opt-in limits for model input bodies, preserving complete table rows."""

import re
from html.parser import HTMLParser


class TokenBudgetError(ValueError):
    """The requested budget cannot contain the mandatory structure and marker."""


class _HTMLTableParser(HTMLParser):
    """Extract complete rows and their header count from the same HTML stream."""

    def __init__(self, body):
        super().__init__()
        self.body = body
        self.line_offsets = [0] + [match.end() for match in re.finditer("\n", body)]
        self.rows = []
        self.column_header_rows = []
        self.row_start = None
        self.row_in_thead = False
        self.has_column_header = False
        self.has_data_cell = False
        self.header_count = None
        self.in_thead = False
        self.template_depth = 0

    def _offset(self):
        line, column = self.getpos()
        return self.line_offsets[line - 1] + column

    def handle_starttag(self, tag, attrs):
        # Template content belongs to a separate document fragment.
        if tag == "template":
            self.template_depth += 1
            return
        if self.template_depth:
            return
        if tag == "thead" and self.header_count is None:
            self.header_count = 0
            self.in_thead = True
        elif tag in {"tbody", "tfoot"}:
            self.in_thead = False
        if tag == "tr" and self.row_start is None:
            self.row_start = self._offset()
            self.row_in_thead = self.in_thead
            self.has_column_header = False
            self.has_data_cell = False
        elif self.row_start is not None:
            if tag == "td":
                self.has_data_cell = True
            elif tag == "th":
                scope = (dict(attrs).get("scope") or "").strip().lower()
                if scope in {"row", "rowgroup"}:
                    self.has_data_cell = True
                else:
                    self.has_column_header = True

    def handle_endtag(self, tag):
        if tag == "template":
            self.template_depth = max(0, self.template_depth - 1)
            return
        if self.template_depth:
            return
        if tag == "tr" and self.row_start is not None:
            end = self.body.find(">", self._offset()) + 1
            self.rows.append(self.body[self.row_start : end])
            self.column_header_rows.append(
                self.has_column_header and not self.has_data_cell
            )
            if self.row_in_thead:
                self.header_count += 1
            self.row_start = None
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
        table = _HTMLTableParser(body)
        table.feed(body)
        table.close()
        rows = table.rows
        if not rows:
            raise TokenBudgetError("HTML table has no complete rows to retain")
        if table.header_count is not None:
            header_count = table.header_count
        else:
            header_count = 0
            for is_header in table.column_header_rows:
                if not is_header:
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
