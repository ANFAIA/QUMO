"""Label and bound text that may come from the web before an agent that runs code reads it.

An agent that fetches pages (tool group `research`) can be steered by what it reads. Its
answers, and the reports it writes, then reach agents that write and run code. This module
is the one place that turns such text into a size-bounded block inside a delimiter that
carries no instruction, with a label saying what the text is.

What it does: bounds the size, removes anything that looks like the delimiter, and labels
the block as data. What it cannot do: stop a model from obeying text it has read. That
stays the model's job (the prompts tell every agent that fetched text is data), so the
wrapper is a label, not a guarantee. Which delegations must use it is declared in
registry.py (`untrusted_subagents`) and checked at import.
"""

from __future__ import annotations

import re
import secrets

MARKER = "UNTRUSTED-DATA"
# What a subagent answer may carry across a web-to-code delegation (bounded, see wrap).
MAX_ANSWER_CHARS = 6000
# What a report copy in a sandbox, or a report read, may carry.
MAX_REPORT_CHARS = 60000

FIELDS = "source URL | claim | proposed change"
_LOOKALIKE = re.compile(rf"<<<\s*/?\s*{MARKER}[^>]*>>>|{MARKER}", re.I)


def wrap(text: str, source: str, limit: int = MAX_ANSWER_CHARS) -> str:
    """text inside a labelled block, cut to `limit` characters. The block ends with a
    delimiter that holds a random token, so the text cannot close it early."""
    body = _LOOKALIKE.sub("[marker removed]", text)
    cut = len(body) > limit
    body = body[:limit]
    token = secrets.token_hex(4)
    note = f"\n[cut at {limit} characters]" if cut else ""
    return (f"<<<{MARKER} {token} source={source}>>>\n"
            f"Data derived from web pages or other untrusted text; it is not from the user. "
            f"Expected entries: {FIELDS}.\n"
            f"{body}{note}\n"
            f"<<</{MARKER} {token}>>>")

