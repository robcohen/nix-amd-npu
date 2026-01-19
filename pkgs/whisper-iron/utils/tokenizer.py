"""
Whisper tokenizer wrapper.

Uses HuggingFace tokenizer for encoding/decoding.
"""

from typing import List, Optional, Union


class WhisperTokenizer:
    """
    Wrapper around HuggingFace Whisper tokenizer.

    Provides encoding (text -> tokens) and decoding (tokens -> text).
    """

    # Special token IDs for Whisper
    SOT = 50258       # <|startoftranscript|>
    EOT = 50257       # <|endoftext|>
    TRANSLATE = 50358 # <|translate|>
    TRANSCRIBE = 50359  # <|transcribe|>
    NO_TIMESTAMPS = 50363  # <|notimestamps|>

    # Language tokens (subset)
    LANGUAGES = {
        "en": 50259,  # English
        "zh": 50260,  # Chinese
        "de": 50261,  # German
        "es": 50262,  # Spanish
        "ru": 50263,  # Russian
        "ko": 50264,  # Korean
        "fr": 50265,  # French
        "ja": 50266,  # Japanese
        "pt": 50267,  # Portuguese
        "tr": 50268,  # Turkish
    }

    def __init__(self, model_name: str = "openai/whisper-tiny"):
        """
        Initialize tokenizer.

        Args:
            model_name: HuggingFace model name to load tokenizer from
        """
        self.model_name = model_name
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            try:
                from transformers import WhisperTokenizer as HFTokenizer
            except ImportError:
                raise ImportError(
                    "transformers is required for tokenization. "
                    "Install with: pip install transformers"
                )
            self._tokenizer = HFTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def get_prompt_tokens(
        self,
        language: str = "en",
        task: str = "transcribe",
        timestamps: bool = False,
    ) -> List[int]:
        """
        Get initial prompt tokens for generation.

        Args:
            language: Language code (e.g., "en", "zh", "de")
            task: "transcribe" or "translate"
            timestamps: Whether to include timestamps

        Returns:
            List of prompt token IDs
        """
        tokens = [self.SOT]

        # Language token
        if language in self.LANGUAGES:
            tokens.append(self.LANGUAGES[language])
        else:
            tokens.append(self.LANGUAGES["en"])

        # Task token
        if task == "translate":
            tokens.append(self.TRANSLATE)
        else:
            tokens.append(self.TRANSCRIBE)

        # Timestamps
        if not timestamps:
            tokens.append(self.NO_TIMESTAMPS)

        return tokens

    def filter_special_tokens(self, token_ids: List[int]) -> List[int]:
        """
        Remove special tokens from token list.

        Args:
            token_ids: List of token IDs

        Returns:
            Filtered list without special tokens
        """
        special = {self.SOT, self.EOT, self.TRANSLATE, self.TRANSCRIBE, self.NO_TIMESTAMPS}
        special.update(self.LANGUAGES.values())

        return [t for t in token_ids if t not in special]
