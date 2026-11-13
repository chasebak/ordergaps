"""docs/ holds one forwarding page since the Gap Explorer was retired.

Why: the interactive Gap Explorer page was retired on 2026-11-13
(DECISIONS.md). Its old address must send every visitor to the Grading Play
Calculator, and nothing of the old page may stay published.

Rules checked:
- docs/ holds exactly .nojekyll and index.html: no page script, style sheet
  or data file is left behind;
- index.html has one meta refresh, with delay 0 and exactly CALCULATOR_URL as
  its target, and one canonical link to the same URL;
- it loads nothing: no script tag, no style sheet link, no src attribute, and
  every link goes to the calculator or to this repository;
- its visible text is the retirement notice and the line back to this
  repository;
- the README states the retirement once and has no "Try it" line.
Failure mode: these checks read the file, they do not open a browser. A
browser that ignores meta refresh shows the notice and its link instead.
Each rule has a control that must fail.
"""

from html.parser import HTMLParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
INDEX_PATH = DOCS_DIR / "index.html"
README_PATH = REPO_ROOT / "README.md"
# Typed here, not read from the page, so the page cannot pass itself.
CALCULATOR_URL = "https://chasebak.github.io/grace-grading-roi/play/"
REPOSITORY_URL = "https://github.com/chasebak/ordergaps"
ALLOWED_DOCS_FILES = [".nojekyll", "index.html"]
# Finder writes this file on macOS; .gitignore keeps it out of the repo.
IGNORED_DOCS_NAMES = (".DS_Store",)
PAGE_NOTICE = (
    "The Gap Explorer was retired on 2026-11-13. "
    "This address now opens the Grading Play Calculator."
)
PAGE_METHOD_LINE = (
    "The method behind the Gap Explorer is in the ordergaps repository on GitHub."
)
README_SENTENCE = (
    "The interactive Gap Explorer page was retired on 2026-11-13; its old address "
    "forwards to the Grading Play Calculator in grace-grading-roi. The last version "
    "with the page is tagged `gap-explorer-final`."
)


class PageParser(HTMLParser):
    """Collects every start tag with its attributes, and the text inside body."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags = []
        self.body_text = []
        self.inside_body = False

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        if tag == "body":
            self.inside_body = True

    def handle_endtag(self, tag):
        if tag == "body":
            self.inside_body = False

    def handle_data(self, data):
        if self.inside_body:
            self.body_text.append(data)


def parse_page(page_text):
    parser = PageParser()
    parser.feed(page_text)
    parser.close()
    return parser


def one_spaced(text):
    """text with every run of whitespace (line breaks too) as one space."""
    words = text.split()
    return " ".join(words)


def refresh_targets(parser):
    """(delay, url) of each meta refresh tag; url is None when it has none.

    Rule: content is "<delay>; url=<target>", the form browsers read.
    """
    targets = []
    for tag, attributes in parser.tags:
        if tag != "meta":
            continue
        http_equiv = attributes.get("http-equiv") or ""
        if http_equiv.lower() != "refresh":
            continue
        content = attributes.get("content") or ""
        delay, _separator, rest = content.partition(";")
        rest = rest.strip()
        url = None
        if rest.lower().startswith("url="):
            url = rest[len("url=") :]
        targets.append((delay.strip(), url))
    return targets


def page_problems(page_text):
    """Reasons page_text is not the forwarding page; empty means it is."""
    parser = parse_page(page_text)
    problems = []

    targets = refresh_targets(parser)
    if targets != [("0", CALCULATOR_URL)]:
        problems.append(f"meta refresh is {targets}")

    canonical_links = []
    allowed_links = (CALCULATOR_URL, REPOSITORY_URL)
    for tag, attributes in parser.tags:
        if tag == "script":
            problems.append("script tag")
        if "src" in attributes:
            problems.append(f"{tag} tag loads a file (src)")
        if tag == "link":
            rel = attributes.get("rel") or ""
            if rel.lower() == "canonical":
                canonical_links.append(attributes.get("href"))
            else:
                problems.append(f"link tag with rel {rel!r}")
        if tag == "a":
            href = attributes.get("href")
            if href not in allowed_links:
                problems.append(f"link to {href}")
    if canonical_links != [CALCULATOR_URL]:
        problems.append(f"canonical link is {canonical_links}")

    visible_text = one_spaced("".join(parser.body_text))
    expected_text = PAGE_NOTICE + " " + PAGE_METHOD_LINE
    if visible_text != expected_text:
        problems.append("visible text is not the retirement notice")
    return problems


def link_targets(page_text):
    """href of every <a> tag, in page order."""
    parser = parse_page(page_text)
    targets = []
    for tag, attributes in parser.tags:
        if tag == "a":
            targets.append(attributes.get("href"))
    return targets


def docs_files(docs_dir):
    """Paths of every file under docs_dir, relative to it, sorted."""
    found = []
    paths = sorted(docs_dir.rglob("*"))
    for path in paths:
        if not path.is_file():
            continue
        if path.name in IGNORED_DOCS_NAMES:
            continue
        relative = path.relative_to(docs_dir)
        found.append(relative.as_posix())
    return found


def test_docs_holds_only_the_forwarding_page():
    assert docs_files(DOCS_DIR) == ALLOWED_DOCS_FILES
    assert (DOCS_DIR / ".nojekyll").read_bytes() == b""


def test_refresh_target_is_exactly_the_calculator_url():
    page_text = INDEX_PATH.read_text(encoding="utf-8")
    parser = parse_page(page_text)
    assert refresh_targets(parser) == [("0", CALCULATOR_URL)]


def test_index_is_a_self_contained_forwarding_page():
    page_text = INDEX_PATH.read_text(encoding="utf-8")
    assert page_problems(page_text) == []
    assert "<script" not in page_text.lower()
    assert link_targets(page_text) == [CALCULATOR_URL, REPOSITORY_URL]


def test_control_a_script_tag_is_caught():
    page_text = INDEX_PATH.read_text(encoding="utf-8")
    planted = page_text.replace("</body>", '<script src="app.js"></script></body>')
    assert planted != page_text
    assert page_problems(planted) == ["script tag", "script tag loads a file (src)"]


def test_control_another_refresh_target_is_caught():
    page_text = INDEX_PATH.read_text(encoding="utf-8")
    old_tag = f'content="0; url={CALCULATOR_URL}"'
    assert page_text.count(old_tag) == 1
    other_target = page_text.replace(old_tag, f'content="0; url={CALCULATOR_URL}index.html"')
    problems = page_problems(other_target)
    assert problems == [f"meta refresh is [('0', '{CALCULATOR_URL}index.html')]"]
    delayed = page_text.replace(old_tag, f'content="5; url={CALCULATOR_URL}"')
    assert page_problems(delayed) == [f"meta refresh is [('5', '{CALCULATOR_URL}')]"]


def test_control_a_style_sheet_link_and_a_changed_notice_are_caught():
    page_text = INDEX_PATH.read_text(encoding="utf-8")
    planted = page_text.replace("</head>", '<link rel="stylesheet" href="tokens.css"></head>')
    assert page_problems(planted) == ["link tag with rel 'stylesheet'"]
    reworded = page_text.replace("retired on 2026-11-13", "retired")
    assert reworded != page_text
    assert page_problems(reworded) == ["visible text is not the retirement notice"]


def test_control_leftover_page_files_are_listed(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (tmp_path / ".nojekyll").write_bytes(b"")
    (tmp_path / "index.html").write_text("<html></html>", encoding="utf-8")
    (tmp_path / "app.js").write_text("", encoding="utf-8")
    (data_dir / "summary.json").write_text("{}", encoding="utf-8")
    found = docs_files(tmp_path)
    assert found == [".nojekyll", "app.js", "data/summary.json", "index.html"]
    assert found != ALLOWED_DOCS_FILES


def test_readme_states_the_retirement_once_and_has_no_try_it_line():
    readme_text = README_PATH.read_text(encoding="utf-8")
    flat_text = one_spaced(readme_text)
    assert flat_text.count(README_SENTENCE) == 1
    assert flat_text.count("Gap Explorer") == 1
    assert "Try it" not in flat_text
