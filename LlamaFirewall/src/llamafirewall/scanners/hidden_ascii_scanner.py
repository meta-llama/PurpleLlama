# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from ..llamafirewall_data_types import (
    Message,
    ScanDecision,
    ScanResult,
    ScanStatus,
    Trace,
)
from .base_scanner import Scanner

# Code points from other Unicode scripts that render identically to (or
# nearly indistinguishably from) an ASCII letter. These are the characters
# most commonly used to disguise keywords via homoglyph substitution. Built
# from the Unicode confusables data (TR39) for ASCII letters A-Za-z. Kept
# compact: only the highest-risk lookalikes (Cyrillic, Greek, Armenian,
# math/full-width already handled via NFKD).
_ASCII_CONFUSABLES = frozenset(
    {
        # A
        0x0410,
        0x0391,
        0x0531,
        0x13AA,
        # a
        0x0430,
        0x03B1,
        0x0561,
        0x0251,
        0x04D1,
        # B
        0x0412,
        0x0392,
        0x0532,
        0x13F4,
        # b (limited high-risk lookalikes)
        0x0432,
        0x03B2,
        0x10192,
        # C
        0x0421,
        0x053E,
        0x042E,
        0x216D,
        # c
        0x0441,
        0x217D,
        0x1D04,
        0x1D18,
        0x03FD,
        # E
        0x0415,
        0x0395,
        0x0535,
        0x13AC,
        # e
        0x0435,
        0x03B5,
        0x04BD,
        0x1D07,
        # H
        0x041D,
        0x0397,
        0x0540,
        0x13BB,
        # K
        0x041A,
        0x039A,
        0x13B6,
        0x212A,
        # k
        0x03BA,
        0x1D0B,
        0x0299,
        # M
        0x041C,
        0x039C,
        0x0544,
        0x13B7,
        0x03FA,
        # O
        0x039F,
        0x0555,
        0x13DE,
        0x047E,
        0x1D32,
        0x2299,
        # o
        0x03BF,
        0x0585,
        0x047F,
        0x1D33,
        0x2134,
        0x1D11,
        # P
        0x03A1,
        0x054C,
        0x13E2,
        0x0420,
        0x1D35,
        # p
        0x03C1,
        0x1D56,
        0x2CA3,
        0x0440,
        0x04AF,
        # b (lowercase Cyrillic в looks like Latin b in some fonts)
        0x0432,
        0x03B2,
        # T
        0x054F,
        0x13A3,
        0x1D40,
        0x0422,
        # y
        0x0443,
        0x1D63,
        0x04AF,
        0x01B4,
        # X
        0x0425,
        0x03A7,
        0x053D,
        0x1D36,
        0x2169,
        # x
        0x0445,
        0x04B3,
        0x03C7,
        0x1D67,
        # I/i l (high-risk for URL/keyword tricks)
        0x0406,
        0x0456,
        0x04CF,
        0x2110,
        0x2111,
        0x1D45,
        0x03B9,
        0x1D2C,
        # J/j
        0x0408,
        0x0458,
        0x1D0A,
        # N/n
        0x039D,
        0x0546,
        0x1D43,
        0x03BD,
        0x1D63,
        # S/s
        0x0405,
        0x0455,
        0x13A7,
        0x01BD,
        0x0283,
        # U/u
        0x054D,
        0x0443,
        0x1D1C,
        0x1D64,
        # V/v
        0x0474,
        0x0475,
        0x1D20,
        0x028B,
        # W/w
        0x0518,
        0x0519,
        0x1D42,
        # Y
        0x03A5,
        0x054E,
        0x1D41,
        0x04AE,
        # Z/z
        0x0396,
        0x0536,
        0x1D3A,
        0x1D68,
        0x0290,
        # g
        0x0261,
        0x018D,
        # d
        0x0501,
        0x010B,
        0x0221,
        0x0257,
        # f
        0x0192,
        0x03DC,
        # r
        0x0280,
        0x0433,
        0x1D63,
        0x027E,
        # t
        0x0442,
        0x1D37,
        0x03C4,
        # 0 (zero, often confused with O)
        0x041E,
        0x0548,
    }
)


class HiddenASCIIScanner(Scanner):
    ALLOW_REASON = "No hidden ASCII detected"

    # Unicode code points used for steganography / hiding ASCII text.
    #
    # The scanner originally checked only the Unicode Tag block
    # (U+E0000..U+E007F), the technique used to embed ASCII invisibly in
    # model output. But "hidden ASCII" in practice spans several other
    # well-documented Unicode techniques that an adversary can use to
    # smuggle instructions past a guard undetected. Each is caught below.
    #
    # References:
    #  - Tag characters: U+E0000..U+E007F (Language Tags block)
    #  - Zero-width chars: U+200B..U+200D, U+2060, U+FEFF (invisible, can
    #    encode binary data between visible characters)
    #  - Bidirectional overrides: U+202A..U+202E (reverse display order to
    #    hide text; U+202E RLO is the classic payload marker)
    #  - Homoglyphs / confusables: characters that render identically to
    #    ASCII letters but have different code points (e.g. Cyrillic "а"
    #    U+0430 vs Latin "a" U+0061), used to disguise keywords.
    TAG_BASE = 0xE0000
    TAG_END = 0xE007F

    ZERO_WIDTH_CHARS = frozenset(
        [
            "\u200b",  # zero-width space
            "\u200c",  # zero-width non-joiner
            "\u200d",  # zero-width joiner
            "\u2060",  # word joiner
            "\ufeff",  # zero-width no-break space / BOM
        ]
    )

    BIDI_OVERRIDE_CHARS = frozenset(
        [
            "\u202a",  # left-to-right embedding
            "\u202b",  # right-to-left embedding
            "\u202c",  # pop directional formatting
            "\u202d",  # left-to-right override
            "\u202e",  # right-to-left override (the common payload marker)
            "\u2066",  # left-to-right isolate
            "\u2067",  # right-to-left isolate
            "\u2068",  # first strong isolate
            "\u2069",  # pop directional isolate
        ]
    )

    def __init__(
        self,
        scanner_name: str = "Hidden ASCII Scanner",
        block_threshold: float = 1.0,
        detect_confusables: bool = True,
    ) -> None:
        super().__init__(scanner_name, block_threshold)
        self.detect_confusables = detect_confusables

    @classmethod
    def _is_hidden_ascii_present(cls, text: str) -> float:
        if cls._has_tag_chars(text):
            return 1.0
        if cls._has_zero_width_chars(text):
            return 1.0
        if cls._has_bidi_overrides(text):
            return 1.0
        return 0.0

    @classmethod
    def _has_tag_chars(cls, text: str) -> bool:
        return any(cls.TAG_BASE <= ord(char) <= cls.TAG_END for char in text)

    @classmethod
    def _has_zero_width_chars(cls, text: str) -> bool:
        return any(char in cls.ZERO_WIDTH_CHARS for char in text)

    @classmethod
    def _has_bidi_overrides(cls, text: str) -> bool:
        return any(char in cls.BIDI_OVERRIDE_CHARS for char in text)

    @staticmethod
    def _decode_hidden_ascii(text: str) -> str:
        base_tag = 0xE0000
        tag_range = range(0xE0000, 0xE007F + 1)
        decoded_text = ""
        for char in text:
            if ord(char) in tag_range:
                decoded_char = chr(ord(char) - base_tag)
                decoded_text += decoded_char
            else:
                decoded_text += char
        return decoded_text

    async def scan(
        self, message: Message, past_trace: Trace | None = None
    ) -> ScanResult:
        text = message.content

        score = self._is_hidden_ascii_present(text)

        # Optional: flag confusable (homoglyph) characters. These don't hide
        # text invisibly but disguise ASCII letters with visually identical
        # lookalikes from other scripts, a common technique for smuggling
        # keywords (e.g. "Ignore" written with a Cyrillic "а"). Kept opt-in
        # because legitimate multilingual content legitimately mixes scripts.
        reasons = []
        if self.detect_confusables and self._has_confusables(text):
            score = max(score, 1.0)
            reasons.append("Confusable/homoglyph characters detected")

        decision = (
            ScanDecision.BLOCK if score >= self.block_threshold else ScanDecision.ALLOW
        )

        if score >= self.block_threshold:
            reason = "Hidden ASCII: " + self._decode_hidden_ascii(text)
            if reasons:
                reason += " | " + "; ".join(reasons)
        else:
            reason = self.ALLOW_REASON

        return ScanResult(
            decision=decision,
            reason=reason,
            score=score,
            status=ScanStatus.SUCCESS,
        )

    @staticmethod
    def _has_confusables(text: str) -> bool:
        """Detect homoglyphs that visually masquerade as ASCII.

        Flags text where a *single token* mixes ASCII Latin letters with
        lookalike characters from another script (e.g. "Ignore" where one
        'a' is Cyrillic U+0430). This is the signature of a homoglyph
        attack: a keyword disguised by substituting a few letters with
        lookalikes. A token that is entirely Cyrillic (e.g. a legitimate
        Russian word) is NOT flagged -- only mixed-script tokens are.

        Full-width Latin letters (U+FF21..U+FF5A) and other compatibility
        confusables follow the same rule: a token of *all* full-width letters
        is legitimate in CJK typography, but a token mixing ASCII Latin with
        full-width lookalikes (e.g. "IGNOR\uff25") is suspicious.
        """
        import unicodedata

        def _is_confusable_letter(char: str) -> bool:
            """True if a non-ASCII alpha char is a Latin-letter lookalike."""
            # Cross-script confusables (Cyrillic/Greek/etc. lookalikes).
            if ord(char) in _ASCII_CONFUSABLES:
                return True
            # Compatibility confusables that decompose to ASCII under NFKD
            # (full-width letters, circled letters, etc.).
            decomposed = unicodedata.normalize("NFKD", char)
            return (
                len(decomposed) == 1 and decomposed.isascii() and decomposed.isalpha()
            )

        # A "token" here is a maximal run of letters. Flag only when a token
        # contains BOTH ASCII Latin letters AND non-ASCII confusables -- that
        # mix is the attack signature. Fully non-ASCII tokens (a legitimate
        # Russian word, or all-full-width caps in Japanese text) are ALLOWed.
        token_has_ascii = False
        token_has_confusable = False
        for char in text:
            if not char.isalpha():
                # Token boundary: reset. We flag as soon as we see both
                # within the same token, so no need to act on completion.
                token_has_ascii = False
                token_has_confusable = False
                continue
            if char.isascii():
                token_has_ascii = True
            elif _is_confusable_letter(char):
                token_has_confusable = True
            if token_has_ascii and token_has_confusable:
                return True
        return False
