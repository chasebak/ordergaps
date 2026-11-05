"""Public privacy check: generic leak patterns in every file git would publish.

Why: this repo is public and runs on synthetic data only. Privacy is checked
in two layers (PROOF_UPGRADE_PLAN.md, "Shared standards" (c) 6):
- this public file holds GENERIC patterns only, so it can sit in a public
  repo without naming what it guards;
- real store, vendor and staff names live in one private deny file outside
  every repo, checked locally before each push. A deny file that lands
  inside the repo is itself a leak, so the last test fails on one.

Scope: tracked files plus untracked files that are not ignored (what the next
commit would pick up). Every published path is scanned too, as line 0 (the
private deny check reports names the same way), binary files included: a
photo named after its cert number publishes that number in its file name.
Binary contents (a NUL byte in the first 8 KiB) are skipped, because random
bytes would match the digit and hex patterns by chance; gitleaks in CI looks
inside them for secrets only, not for cert numbers or home paths, so keep
binaries out of a public repo unless they are generated from synthetic data.
Failure output names file, line and pattern only, never the matched text, so
a public CI log cannot repeat the leak it caught.
Failure mode: these patterns miss a leak that has no generic shape (a real
name in plain words); that is what the private deny check is for.
The planted examples in the control tests are built from pieces, so this file
never matches its own patterns.
"""

import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BINARY_SNIFF_BYTES = 8192

# Built from pieces so this file does not contain the path prefix it bans.
HOME_PATH_PATTERN = re.compile("/" + "Users" + "/")
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
HEX64_PATTERN = re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{64}(?![0-9A-Fa-f])")
# An 8-10 digit run with digit-only boundaries: letters or "_" next to it do
# not hide it, because a cert is often written that way (a grader prefix glued
# on, or a photo named "<grader>_<cert>.jpg"); a \w boundary let both through.
# Still excluded: longer digit runs, the fraction of a decimal ("0.1234...")
# and the whole part of one.
CERT_PATTERN = re.compile(r"(?<![\d.])\d{8,10}(?!\d|\.\d)")
NOREPLY_DOMAIN = "users.noreply.github.com"


def git_paths(extra_args):
    """Repo-relative paths from `git ls-files -z` with the given flags.

    Why -z: synthetic data files have spaces in their names; NUL-separated
    output keeps them whole.
    """
    command = ["git", "-C", str(REPO_ROOT), "ls-files", "-z", *extra_args]
    completed = subprocess.run(command, capture_output=True, check=True)
    output = completed.stdout.decode("utf-8")
    raw_paths = output.split("\0")
    paths = []
    for raw_path in raw_paths:
        if raw_path:
            paths.append(raw_path)
    return paths


def is_notebook_image_line(path, line):
    """True for a base64 image payload line inside a Jupyter notebook.

    Why: an embedded PNG is a long random string that can contain a digit
    run or a hex run by chance; it carries no text a person wrote.
    """
    if not path.endswith(".ipynb"):
        return False
    stripped = line.lstrip()
    return stripped.startswith('"image/')


def read_published_lines():
    """(path, line number, line) for every path and text line git would publish.

    Each path comes first as line 0 with the path itself as the text, before
    the binary skip, so a cert number or e-mail in a file name is caught
    even when the file's contents are never read.
    """
    tracked = git_paths(["--cached"])
    untracked = git_paths(["--others", "--exclude-standard"])
    all_paths = sorted(set(tracked) | set(untracked))
    lines = []
    for path in all_paths:
        lines.append((path, 0, path))
        full_path = REPO_ROOT / path
        if not full_path.is_file():
            continue
        data = full_path.read_bytes()
        head = data[:BINARY_SNIFF_BYTES]
        if b"\0" in head:
            continue
        text = data.decode("utf-8", errors="replace")
        text_lines = text.splitlines()
        numbered = enumerate(text_lines, start=1)
        for line_number, line in numbered:
            if is_notebook_image_line(path, line):
                continue
            lines.append((path, line_number, line))
    return lines


@pytest.fixture(scope="module")
def published_lines():
    return read_published_lines()


def author_emails():
    """Author addresses declared in pyproject.toml, lowercased.

    Why: the author's own address is public by choice (LICENSE, package
    metadata); reading it from pyproject.toml keeps it out of this file and
    allows exactly that address, not a whole domain.
    """
    pyproject_path = REPO_ROOT / "pyproject.toml"
    if not pyproject_path.is_file():
        return set()
    pyproject_text = pyproject_path.read_text(encoding="utf-8")
    pyproject = tomllib.loads(pyproject_text)
    project = pyproject.get("project", {})
    authors = project.get("authors", [])
    emails = set()
    for author in authors:
        email = author.get("email", "")
        if email:
            emails.add(email.lower())
    return emails


def email_is_allowed(address, allowed_emails):
    """The author's own address and GitHub noreply addresses are allowed."""
    lowered = address.lower()
    if lowered in allowed_emails:
        return True
    return lowered.endswith("@" + NOREPLY_DOMAIN)


def find_hits(lines, pattern, pattern_name, allow=None):
    """'file:line: pattern' for each line where pattern matches (minus allowed).

    The matched text is left out on purpose: a public CI log must not repeat
    the leak it caught.
    """
    hits = []
    for path, line_number, line in lines:
        matches = pattern.findall(line)
        for match in matches:
            if allow is not None and allow(match):
                continue
            hits.append(f"{path}:{line_number}: {pattern_name}")
            break
    return hits


def deny_named_paths(paths):
    """Paths whose file name starts with 'deny' (any case)."""
    found = []
    for path in paths:
        name = Path(path).name.lower()
        if name.startswith("deny"):
            found.append(path)
    return found


def test_no_home_directory_paths(published_lines):
    """An absolute macOS home path leaks the author's account name and local layout."""
    hits = find_hits(published_lines, HOME_PATH_PATTERN, "home-directory path")
    assert hits == []


def test_no_email_addresses(published_lines):
    """An email address identifies a real person or account; only the author's own is allowed."""
    allowed_emails = author_emails()

    def allow(address):
        return email_is_allowed(address, allowed_emails)

    hits = find_hits(published_lines, EMAIL_PATTERN, "email address", allow=allow)
    assert hits == []


def test_no_64_hex_strings(published_lines):
    """A 64-hex run is shaped like a store admin persisted-query hash or an API secret."""
    hits = find_hits(published_lines, HEX64_PATTERN, "64-hex string")
    assert hits == []


def test_no_cert_like_digit_runs(published_lines):
    """An 8-10 digit run, even glued to letters, is shaped like a real grading-cert number.

    The portfolio's synthetic ids never take that shape: SYN-nnnnnn certs and
    7-digit product ids from 9,000,000 (cardshop-agents), card_NNNN
    (grace-grading-roi), csyn/gsyn ids (cardsegments), 6-digit product ids
    (milo). Decimals are excluded, because simulated results print with
    many decimal places.
    """
    hits = find_hits(published_lines, CERT_PATTERN, "cert-like digit run")
    assert hits == []


def test_no_deny_file_in_repo():
    """A deny* file inside the repo would publish the private names it exists to guard.

    Ignored files count too (--others without --exclude-standard): an
    ignored deny file is one `git add -f` away from being public.
    """
    tracked = git_paths(["--cached"])
    on_disk = git_paths(["--others"])
    all_paths = sorted(set(tracked) | set(on_disk))
    found = deny_named_paths(all_paths)
    assert found == []


# Controls: each check must be able to fail, so each pattern is shown to catch
# a planted example and to pass the synthetic shapes that sit beside it.


def planted_lines(text):
    return [("planted.txt", 1, text)]


def test_control_home_path_matches_planted_example():
    planted = "see " + "/" + "Users" + "/someone/project/data.csv"
    hits = find_hits(planted_lines(planted), HOME_PATH_PATTERN, "home-directory path")
    assert hits == ["planted.txt:1: home-directory path"]
    clean = find_hits(planted_lines("data/synthetic/users.csv"), HOME_PATH_PATTERN, "x")
    assert clean == []


def test_control_email_matches_planted_example_and_allows_noreply():
    planted = "contact " + "someone" + "@" + "example.org" + " today"
    hits = find_hits(planted_lines(planted), EMAIL_PATTERN, "email address")
    assert hits == ["planted.txt:1: email address"]
    noreply = "12345+bot" + "@" + NOREPLY_DOMAIN
    assert email_is_allowed(noreply, set())
    author = "author" + "@" + "example.org"
    assert email_is_allowed(author.upper(), {author})
    other = "someone" + "@" + "example.org"
    assert not email_is_allowed(other, {author})


def test_control_hex64_matches_planted_example_only_at_64():
    planted = "hash=" + "ab12" * 16
    hits = find_hits(planted_lines(planted), HEX64_PATTERN, "64-hex string")
    assert hits == ["planted.txt:1: 64-hex string"]
    short = "hash=" + ("ab12" * 16)[:63]
    long_run = "hash=" + "ab12" * 16 + "c"
    assert find_hits(planted_lines(short), HEX64_PATTERN, "x") == []
    assert find_hits(planted_lines(long_run), HEX64_PATTERN, "x") == []


def test_control_cert_pattern_matches_planted_examples():
    planted_runs = ["1234" + "5678", "1234" + "56789", "1234" + "567890"]
    for run in planted_runs:
        planted = "cert " + run + " graded 10"
        hits = find_hits(planted_lines(planted), CERT_PATTERN, "cert-like digit run")
        assert hits == ["planted.txt:1: cert-like digit run"]


def test_control_cert_pattern_matches_runs_glued_to_letters():
    """A grader prefix or "_" next to the digits must not hide a cert."""
    run = "1234" + "5678"
    glued_texts = ["PSA" + run, "psa_" + run + ".jpg", "cert" + run + "x"]
    for text in glued_texts:
        hits = find_hits(planted_lines(text), CERT_PATTERN, "cert-like digit run")
        assert hits == ["planted.txt:1: cert-like digit run"], text


def test_control_published_paths_are_scanned_as_line_zero(tmp_path, monkeypatch):
    """A binary photo named after its cert is caught by its path alone.

    Plants a throwaway git repo so the real scan function runs end to end:
    the image's contents are skipped as binary, its name still reports.
    """
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    run = "1234" + "5678"
    photo_name = "psa_" + run + ".jpg"
    photo_path = tmp_path / "photos" / photo_name
    photo_path.parent.mkdir()
    photo_path.write_bytes(b"\xff\xd8\0binary")
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    lines = read_published_lines()
    assert lines == [("photos/" + photo_name, 0, "photos/" + photo_name)]
    hits = find_hits(lines, CERT_PATTERN, "cert-like digit run")
    assert hits == ["photos/" + photo_name + ":0: cert-like digit run"]


def test_control_cert_pattern_passes_synthetic_shapes():
    synthetic_texts = [
        "cert SYN-" + "123456",
        "product 9000001",
        "card_0042",
        "csyn000000,gsyn0007",
        "share 0.1234" + "5678",
        "total 1234" + "5678.5",
        "epoch 1234" + "5678901",
    ]
    for text in synthetic_texts:
        hits = find_hits(planted_lines(text), CERT_PATTERN, "cert-like digit run")
        assert hits == [], text


def test_control_deny_file_name_check_catches_planted_paths():
    paths = ["data/deny.txt", "Denylist.csv", "tests/test_x.py", "docs/undeny.md"]
    found = deny_named_paths(paths)
    assert found == ["data/deny.txt", "Denylist.csv"]


def test_control_notebook_image_lines_are_the_only_skipped_lines():
    image_line = '      "image/png": "iVBORw0KGgo"'
    assert is_notebook_image_line("walkthrough.ipynb", image_line)
    assert not is_notebook_image_line("README.md", image_line)
    assert not is_notebook_image_line("walkthrough.ipynb", '    "print(1)"')
