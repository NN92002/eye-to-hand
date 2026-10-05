#!/usr/bin/env python3
"""Keep every package.xml <version> equal to the repo-root VERSION file.

The repository is versioned as a whole: one version for all packages, one tag
per release. package.xml has to carry a literal version string — ament, rosdep
and bloom parse it statically, so it cannot reference a variable — which means
the version is necessarily duplicated. VERSION is the authoritative copy and
this script is what propagates it; the --check mode runs in pre-commit and CI so
the copies cannot drift.

    dev/version.py                 # report the current version and any drift
    dev/version.py --check         # exit non-zero on drift (pre-commit, CI)
    dev/version.py --set 1.2.0     # write VERSION and every package.xml
    dev/version.py --check-tag v1.1.0   # assert a release tag matches VERSION
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VERSION_FILE = REPO / "VERSION"

# <version> appears once per package.xml, and only the first occurrence is the
# package's own version, so the substitution is deliberately count-limited.
VERSION_TAG = re.compile(r"(<version>)([^<]*)(</version>)")
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def package_xmls():
    """Every package.xml we own.

    Enumerated through git rather than a glob: tracked files only, so colcon
    build/install trees carrying copies of package.xml stay out, and submodule
    contents (gitlinks, not trees) are excluded for free.
    """
    listed = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "*package.xml"],
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(REPO / line for line in listed.stdout.split())


def read_version():
    """The version from VERSION: first line that is not blank or a comment.

    The version stays on line 1 so `head -1 VERSION` remains a valid way to read
    it; the comment block listing the package.xml copies follows below.
    """
    for line in VERSION_FILE.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    raise SystemExit(f"{VERSION_FILE} holds no version line")


def listed_package_xmls():
    """The package.xml paths named in VERSION's reminder block."""
    listed = []
    for line in VERSION_FILE.read_text().splitlines():
        stripped = line.lstrip("#").strip()
        if stripped.endswith("package.xml"):
            listed.append(stripped)
    return sorted(listed)


def render_version_file(version):
    """VERSION's full contents: the version, then the generated reminder.

    The list is generated rather than hand-kept, and --check compares it against
    the tracked package.xml files, so it cannot quietly go stale when a package
    is added or removed.
    """
    lines = [
        version,
        "",
        "# Authoritative version for the whole repository. Every package.xml below",
        "# carries a copy, because ament, rosdep and bloom parse that version",
        "# statically and it cannot reference a variable:",
        "#",
    ]
    lines += [f"#   {p.relative_to(REPO)}" for p in package_xmls()]
    lines += [
        "#",
        "# Bump with `dev/version.py --set X.Y.Z`, which rewrites this file and every",
        "# package.xml above. A pre-commit hook fails if they drift, or if this list",
        "# stops matching the packages in the repo.",
    ]
    return "\n".join(lines) + "\n"


def package_version(path):
    match = VERSION_TAG.search(path.read_text())
    return match.group(2) if match else None


def drift(expected):
    return [
        (p.relative_to(REPO), package_version(p))
        for p in package_xmls()
        if package_version(p) != expected
    ]


def write_version(path, version):
    text = path.read_text()
    updated = VERSION_TAG.sub(rf"\g<1>{version}\g<3>", text, count=1)
    if updated != text:
        path.write_text(updated)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="fail on drift")
    group.add_argument("--set", metavar="X.Y.Z", help="set the repo version")
    group.add_argument("--check-tag", metavar="TAG", help="assert TAG matches VERSION")
    args = parser.parse_args()

    if args.set:
        if not SEMVER.match(args.set):
            parser.error(f"not a semantic version: {args.set}")
        VERSION_FILE.write_text(render_version_file(args.set))
        for path in package_xmls():
            if write_version(path, args.set):
                print(f"updated {path.relative_to(REPO)}")
        print(f"repo version is now {args.set}")
        return 0

    expected = read_version()

    if args.check_tag:
        # Release tags are the repo version with a leading v.
        if args.check_tag.lstrip("vV") != expected:
            print(
                f"tag {args.check_tag} does not match VERSION ({expected})",
                file=sys.stderr,
            )
            return 1
        print(f"tag {args.check_tag} matches VERSION")
        return 0

    tracked = [str(p.relative_to(REPO)) for p in package_xmls()]
    stale = listed_package_xmls() != sorted(tracked)
    mismatched = drift(expected)

    if not mismatched and not stale:
        print(f"{expected}: {len(tracked)} package.xml files agree")
        return 0

    if stale:
        listed = set(listed_package_xmls())
        print("VERSION's package.xml list no longer matches the repo:", file=sys.stderr)
        for path in sorted(listed ^ set(tracked)):
            side = "listed, no longer in the repo" if path in listed else "not listed"
            print(f"  {path}: {side}", file=sys.stderr)

    if mismatched:
        print(f"VERSION says {expected}, but:", file=sys.stderr)
        for path, found in mismatched:
            print(f"  {path}: {found}", file=sys.stderr)
    if args.check:
        print("run dev/version.py --set <version> to fix", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
