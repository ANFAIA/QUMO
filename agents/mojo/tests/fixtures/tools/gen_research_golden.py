"""Write research_golden.json: what Python's research tools return for fixed inputs.

Dev-time only (owner T, DESIGN.md 8.2): test_tools_atom_html compares tools/atom.mojo and
tools/html.mojo with these outputs. Network is never used: research._get is replaced by a
function returning the fixture bytes.

    agents/.venv/bin/python -I agents/mojo/tests/fixtures/tools/gen_research_golden.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "agents"))

from qumo_agents.tools import research  # noqa: E402

A = 'xmlns="http://www.w3.org/2005/Atom"'

ATOM_CASES = {
    "test_agents_sample": f"""<feed {A}><entry>
<id>http://arxiv.org/abs/2610.00001v1</id><title>A  routing
method</title><published>2026-10-01T00:00:00Z</published>
<summary>We route.</summary><author><name>A. B.</name></author></entry></feed>""",
    "arxiv_like": f"""<?xml version="1.0" encoding="UTF-8"?>
<feed {A} xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <link href="http://arxiv.org/api/query?search_query%3Dall%3Arouting" rel="self" type="application/atom+xml"/>
  <title type="html">ArXiv Query: search_query=all:routing&amp;id_list=</title>
  <id>http://arxiv.org/api/abc</id>
  <updated>2026-11-06T00:00:00-05:00</updated>
  <opensearch:totalResults>2</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2611.01234v2</id>
    <updated>2026-11-05T17:59:59Z</updated>
    <published>2026-11-04T12:00:00Z</published>
    <title>Qubit Routing with\n  Lookahead &amp; Swaps: a &lt;fast&gt; method</title>
    <summary>  We present a router.\nIt is fast &#x2014; really&#33;\r\n  Second line.
</summary>
    <author><name>Ana Gómez</name><arxiv:affiliation>Univ</arxiv:affiliation></author>
    <author><name>Bo Li</name></author>
    <arxiv:doi>10.1000/xyz</arxiv:doi>
    <link title="doi" href="http://dx.doi.org/10.1000/xyz" rel="related"/>
    <arxiv:primary_category term="quant-ph" scheme="http://arxiv.org/schemas/atom"/>
    <category term="quant-ph" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2611.00001v1</id>
    <published>2026-11-01T00:00:00Z</published>
    <title><![CDATA[CDATA <title> & more]]></title>
    <summary>{"long summary words " * 80}</summary>
    {"".join(f"<author><name>Author {i}</name></author>" for i in range(8))}
  </entry>
</feed>
""",
    "prefixed_and_odd": """<!-- leading comment --><?pi data?>
<a:feed xmlns:a="http://www.w3.org/2005/Atom" xmlns:o="urn:other">
<a:entry>
  <a:id>no-slash-id</a:id>
  <a:title>Before<!-- c -->After<a:sub>child text</a:sub>tail</a:title>
  <a:published>2026</a:published>
  <a:author><a:email>x@y</a:email></a:author>
  <a:author><o:name>wrong ns</o:name><a:name></a:name></a:author>
  <a:author><a:name>Real &#233;&#x1F600;</a:name></a:author>
</a:entry>
<o:entry><a:id>ignored, other namespace</a:id></o:entry>
<a:entry>
  <a:title>No id here</a:title>
  <a:summary/>
</a:entry>
<entry><a:id>unprefixed entry is not Atom here</a:id></entry>
<a:entry xmlns:a="urn:not-atom"><a:id>rebound prefix</a:id></a:entry>
<a:entry><x xmlns="http://www.w3.org/2005/Atom"><id>nested, not direct</id></x><id xmlns="http://www.w3.org/2005/Atom">http://h/default-ns-id</id></a:entry>
</a:feed>
""",
    "bom_doctype_crlf": "﻿<?xml version=\"1.0\"?>\r\n<!DOCTYPE feed [<!ELEMENT feed ANY>]>\r\n"
                        f"<feed {A}>\r\n<entry><id>http://x/1</id><title>T\r\nU\rV</title>"
                        "<summary>a&#13;b\r\nc</summary></entry></feed>\r\n",
    "empty_feed": f"<feed {A}/>",
    "err_mismatched": f"<feed {A}><entry><id>x</entry></feed>",
    "err_mismatched_line2": f"<feed {A}>\n  <éntry></entry></feed>",
    "err_no_element": f"<feed {A}><entry>",
    "err_empty": "",
    "err_undefined_entity": f"<feed {A}><entry><title>&nbsp;</title></entry></feed>",
    "err_junk": f"<feed {A}></feed><feed/>",
    "err_unbound_prefix": "<feed><x:entry/></feed>",
    "err_duplicate_attribute": f'<feed {A} a="1" a="2"/>',
    "err_bad_amp": f"<feed {A}>a & b</feed>",
    "err_bad_char_ref": f"<feed {A}>&#1;</feed>",
}

HTML_CASES = {
    "basic": "<html><head><title>T</title><style>body { color: red }</style></head>"
             "<body><h1>Hello</h1><p>World &amp; more</p><script>var x = '<b>';</script></body></html>",
    "upper_and_attrs": "<SCRIPT type=\"text/javascript\">a()</SCRIPT>keep<Style media=x>p{}</STYLE>this",
    "multiline_and_shortest": "<script>\none\n</script>mid<script>two</script>end",
    "unclosed_script": "<script>never closed <b>bold</b> text",
    "prefix_names": "<scripts>gone</scripts>x<styles>also</styles>y<scriptx>z</scriptx>w<script >v</script >u",
    "unicode_ignorecase": "<ſcript>a</ſcript>b<scrİpt>c</scrİpt>d<scrıpt>e</script>f<SCRİPT>g</script>h",
    "angles": "a < b > c <> d <<e>> f < g",
    "whitespace": "x　y z w\tv\n\nu\x1cq  <br/>  r",
    "comment_tags": "<!-- a comment --><p>Text</p><!-- <script> -->x<script>y</script>",
    "long": "<p>" + " ".join(f"w{i}" for i in range(5000)) + "</p>",
}


def atom_result(xml: str) -> str:
    research._get = lambda url, headers=None: xml.encode()
    try:
        return research._json(research.search_arxiv_raw("q", ("quant-ph",), 10))
    except research.NET_ERRORS as e:
        return f"{type(e).__name__}: {e}"


def html_result(html: str) -> str:
    research._get = lambda url, headers=None: html.encode()
    return research.fetch_page.invoke({"url": "https://example.org/"})


out = {"schema": "qumo.test_research_golden.v1",
       "atom": [{"name": n, "xml": x, "expected": atom_result(x)} for n, x in ATOM_CASES.items()],
       "html": [{"name": n, "html": h, "expected": html_result(h)} for n, h in HTML_CASES.items()]}
(HERE / "research_golden.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
