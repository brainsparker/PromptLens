"""Replace personal data in imported text with typed placeholders.

Golden sets get committed to Git and pasted into judge prompts, so production
inputs are scrubbed before they are written. The scrubber is deliberately
conservative: it targets formats that are unambiguous (email addresses, card
numbers that pass Luhn, IP addresses, US social security and phone numbers)
and reports what it replaced so a reviewer can double check.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Pattern, Tuple

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
IPV4_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
SSN_RE = re.compile(r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
PHONE_RE = re.compile(
    r"(?<![\w.-])(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?![\w-])"
)
CARD_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
URL_CREDENTIAL_RE = re.compile(r"(https?://)[^\s/@:]+:[^\s/@]+@")
SECRET_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|sk-ant-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,})\b"
)


def _luhn_ok(digits: str) -> bool:
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        number = int(char)
        if index % 2 == parity:
            number *= 2
            if number > 9:
                number -= 9
        total += number
    return total % 10 == 0


def _card_replacer(match: "re.Match[str]") -> str:
    digits = re.sub(r"\D", "", match.group(0))
    if 13 <= len(digits) <= 19 and _luhn_ok(digits):
        return "[CARD]"
    return match.group(0)


@dataclass
class ScrubReport:
    """Counts of replacements made, by placeholder."""

    replacements: Dict[str, int] = field(default_factory=dict)

    def add(self, placeholder: str, count: int) -> None:
        if count:
            self.replacements[placeholder] = self.replacements.get(placeholder, 0) + count

    @property
    def total(self) -> int:
        return sum(self.replacements.values())

    def merge(self, other: "ScrubReport") -> None:
        for placeholder, count in other.replacements.items():
            self.add(placeholder, count)


# Order matters: secrets and emails first so their digits and dots are not
# re-matched by the looser phone and card patterns.
_RULES: List[Tuple[str, Pattern[str]]] = [
    ("[SECRET]", SECRET_RE),
    ("[EMAIL]", EMAIL_RE),
    ("[SSN]", SSN_RE),
    ("[IP]", IPV4_RE),
]


def scrub_text(
    text: Optional[str], extra_patterns: Optional[List[Pattern[str]]] = None
) -> Tuple[Optional[str], ScrubReport]:
    """Scrub one string. Returns the new text and a report of replacements."""
    report = ScrubReport()
    if not text:
        return text, report

    scrubbed = text
    scrubbed, count = URL_CREDENTIAL_RE.subn(r"\1[CREDENTIAL]@", scrubbed)
    report.add("[CREDENTIAL]", count)

    for placeholder, pattern in _RULES:
        scrubbed, count = pattern.subn(placeholder, scrubbed)
        report.add(placeholder, count)

    before = scrubbed
    scrubbed = CARD_RE.sub(_card_replacer, scrubbed)
    report.add("[CARD]", scrubbed.count("[CARD]") - before.count("[CARD]"))

    scrubbed, count = PHONE_RE.subn("[PHONE]", scrubbed)
    report.add("[PHONE]", count)

    for pattern in extra_patterns or []:
        scrubbed, count = pattern.subn("[REDACTED]", scrubbed)
        report.add("[REDACTED]", count)

    return scrubbed, report
