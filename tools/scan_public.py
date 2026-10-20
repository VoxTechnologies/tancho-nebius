"""Refuse to publish a tree that still names a client, a cloud resource id, or a credential."""
import argparse
import os
import pathlib
import re
import sys

# Project-specific forbidden terms (client names, product code names) stay out of this file,
# which is itself published. Put them one per line in the file named by SCAN_TERMS_FILE,
# or in tools/scan-terms.local (gitignored).
PATTERNS = {
    'cloud resource id': r'aijob-[a-z0-9]+|project-e00[a-z0-9]+|computefilesystem-[a-z0-9]+|registry-e00[a-z0-9]+|ap-[a-z0-9]{12,}',
    'local absolute path': r'/Users/[A-Za-z0-9._-]+',
    'credential': r'-----BEGIN [A-Z ]*PRIVATE KEY|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|api[_-]?key\s*=\s*[\'"][^\'"]{16,}',
    'internal doc reference': r'\.\./\.\./|Projects/[A-Z]',
}
SKIP_DIRS = {'.git', '__pycache__', '.venv', 'runs'}
TEXT_SUFFIXES = {'.py', '.md', '.json', '.txt', '.yml', '.yaml', '.sh', '.toml', '.cfg'}


def extra_terms():
    path = os.environ.get('SCAN_TERMS_FILE') or (pathlib.Path(__file__).parent / 'scan-terms.local')
    path = pathlib.Path(path)
    if not path.is_file():
        return {}
    terms = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return {'forbidden term': '|'.join(re.escape(term) for term in terms)} if terms else {}


def scan(root):
    patterns = {**PATTERNS, **extra_terms()}
    findings = []
    for path in sorted(pathlib.Path(root).rglob('*')):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        if path.name == 'scan_public.py':
            continue
        text = path.read_text(errors='replace')
        for label, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                line = text.count('\n', 0, match.start()) + 1
                findings.append((str(path.relative_to(root)), line, label, match.group()[:60]))
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=pathlib.Path)
    root = parser.parse_args().root
    findings = scan(root)
    for row in findings:
        print('{}:{}: {}: {}'.format(*row))
    print('{} finding(s) over {} file(s)'.format(
        len(findings), sum(1 for p in root.rglob('*') if p.is_file())))
    return 1 if findings else 0


if __name__ == '__main__':
    sys.exit(main())
