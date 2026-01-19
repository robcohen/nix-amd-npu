"""
Full Whisper model combining encoder and decoder.
"""

import numpy as np
from typing import Optional, List

from .config import WhisperConfig, WHISPER_TINY
from .encoder import WhisperEncoder
from .decoder import WhisperDecoder

BF16_DTYPE = np.float32


class Whisper:
    """
    Complete Whisper speech-to-text model.

    Combines:
    - Audio encoder (mel spectrogram -> embeddings)
    - Text decoder (embeddings + tokens -> logits)
    """

    def __init__(self, config: WhisperConfig = WHISPER_TINY):
        self.config = config
        self.encoder = WhisperEncoder(config)
        self.decoder = WhisperDecoder(config)

    def encode(self, mel: np.ndarray) -> np.ndarray:
        """
        Encode audio to embeddings.

        Args:
            mel: Mel spectrogram of shape (batch, n_mels, n_frames)

        Returns:
            Encoder output of shape (batch, seq_len, d_model)
        """
        return self.encoder(mel)

    def decode(
        self,
        input_ids: np.ndarray,
        encoder_output: np.ndarray,
    ) -> np.ndarray:
        """
        Decode tokens to logits.

        Args:
            input_ids: Token IDs of shape (batch, tgt_len)
            encoder_output: Encoder output from encode()

        Returns:
            Logits of shape (batch, tgt_len, vocab_size)
        """
        return self.decoder(input_ids, encoder_output)

    def forward(
        self,
        mel: np.ndarray,
        input_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Full forward pass.

        Args:
            mel: Mel spectrogram of shape (batch, n_mels, n_frames)
            input_ids: Token IDs of shape (batch, tgt_len)

        Returns:
            Logits of shape (batch, tgt_len, vocab_size)
        """
        encoder_output = self.encode(mel)
        logits = self.decode(input_ids, encoder_output)
        return logits

    def generate(
        self,
        mel: np.ndarray,
        max_length: int = 448,
        start_token_id: int = 50258,  # <|startoftranscript|>
        end_token_id: int = 50257,    # <|endoftext|>
        language_token_id: int = 50259,  # <|en|>
        transcribe_token_id: int = 50359,  # <|transcribe|>
        no_timestamps_token_id: int = 50363,  # <|notimestamps|>
    ) -> List[int]:
        """
        Generate transcription using greedy decoding.

        Args:
            mel: Mel spectrogram of shape (1, n_mels, n_frames)
            max_length: Maximum output length
            start_token_id: Start of transcript token
            end_token_id: End of text token
            language_token_id: Language token (English)
            transcribe_token_id: Task token (transcribe vs translate)
            no_timestamps_token_id: Disable timestamps

        Returns:
            List of generated token IDs
        """
        # Encode audio once
        encoder_output = self.encode(mel)

        # Initialize with special tokens
        # <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
        tokens = [start_token_id, language_token_id, transcribe_token_id, no_timestamps_token_id]

        # Greedy decoding
        for _ in range(max_length - len(tokens)):
            # Get input IDs
            input_ids = np.array([tokens], dtype=np.int64)

            # Forward pass
            logits = self.decode(input_ids, encoder_output)

            # Get next token (greedy: argmax of last position)
            next_token = int(np.argmax(logits[0, -1, :]))

            # Stop if end token
            if next_token == end_token_id:
                break

            tokens.append(next_token)

        return tokens

    def load_weights(self, state_dict: dict):
        """
        Load weights from HuggingFace state dict.

        Args:
            state_dict: Full model state dict
        """
        # Split state dict for encoder and decoder
        encoder_state = {}
        decoder_state = {}

        for key, value in state_dict.items():
            if key.startswith("model.encoder."):
                new_key = key.replace("model.encoder.", "")
                encoder_state[new_key] = value
            elif key.startswith("model.decoder."):
                new_key = key.replace("model.decoder.", "")
                decoder_state[new_key] = value

        self.encoder.load_weights(encoder_state)
        self.decoder.load_weights(decoder_state)

    @classmethod
    def from_pretrained(cls, model_name: str = "openai/whisper-tiny") -> "Whisper":
        """
        Load pretrained Whisper model from HuggingFace.

        Args:
            model_name: HuggingFace model name

        Returns:
            Loaded Whisper model
        """
        # Import here to avoid dependency if not using pretrained
        try:
            from transformers import WhisperForConditionalGeneration
        except ImportError:
            raise ImportError(
                "transformers is required to load pretrained weights. "
                "Install with: pip install transformers"
            )

        # Determine config based on model name
        if "tiny" in model_name:
            config = WHISPER_TINY
        elif "base" in model_name:
            from .config import WHISPER_BASE
            config = WHISPER_BASE
        elif "small" in model_name:
            from .config import WHISPER_SMALL
            config = WHISPER_SMALL
        else:
            config = WHISPER_TINY

        # Create model
        model = cls(config)

        # Load HuggingFace model
        print(f"Loading {model_name} from HuggingFace...")
        hf_model = WhisperForConditionalGeneration.from_pretrained(model_name)

        # Convert state dict to numpy
        state_dict = {
            key: value.detach().cpu().numpy()
            for key, value in hf_model.state_dict().items()
        }

        # Load weights
        model.load_weights(state_dict)
        print("Weights loaded successfully!")

        return model
