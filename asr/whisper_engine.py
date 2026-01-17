"""
ASR Engine using OpenAI Whisper for audio transcription.
"""

import os
from typing import Tuple, Optional
import warnings

try:
    import whisper
except ImportError:
    raise ImportError("Whisper not installed. Run: pip install openai-whisper")


class WhisperEngine:
    """
    ASR Engine for transcribing audio using OpenAI Whisper.
    """
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        """
        Initialize Whisper engine.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to use ('cuda' or 'cpu'). If None, auto-detect.
        """
        self.model_size = model_size
        self.device = device
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Whisper model with error handling."""
        try:
            # Suppress FP16 warnings
            warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
            
            self.model = whisper.load_model(
                self.model_size,
                device=self.device
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Whisper model: {str(e)}")
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Tuple[str, float]:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to the audio file
            language: Language code (e.g., 'en', 'es'). If None, auto-detect.
            
        Returns:
            Tuple of (transcribed_text, average_confidence)
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio format is not supported
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Transcribe with Whisper
            result = self.model.transcribe(
                audio_path,
                language=language,
                fp16=False  # Disable FP16 for CPU compatibility
            )
            
            # Extract text
            text = result.get("text", "").strip()
            
            # Calculate average confidence from segments
            segments = result.get("segments", [])
            confidences = []
            
            for segment in segments:
                # Whisper provides "no_speech_prob" - convert to confidence
                no_speech_prob = segment.get("no_speech_prob", 0.0)
                confidence = 1.0 - no_speech_prob
                confidences.append(confidence)
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.8
            
            return text, avg_confidence
            
        except Exception as e:
            raise ValueError(f"Error transcribing audio: {str(e)}")
    
    def transcribe_with_timestamps(self, audio_path: str, language: Optional[str] = None) -> dict:
        """
        Transcribe audio with word-level timestamps.
        
        Args:
            audio_path: Path to the audio file
            language: Language code. If None, auto-detect.
            
        Returns:
            Dictionary containing full transcription and segments with timestamps
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            result = self.model.transcribe(
                audio_path,
                language=language,
                fp16=False,
                word_timestamps=True
            )
            
            return {
                "text": result.get("text", "").strip(),
                "language": result.get("language", "unknown"),
                "segments": result.get("segments", [])
            }
            
        except Exception as e:
            raise ValueError(f"Error transcribing audio with timestamps: {str(e)}")
    
    def detect_language(self, audio_path: str) -> Tuple[str, float]:
        """
        Detect the language of the audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple of (language_code, confidence)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Load audio
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            
            # Make log-Mel spectrogram
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            
            # Detect language
            _, probs = self.model.detect_language(mel)
            
            # Get most likely language
            detected_lang = max(probs, key=probs.get)
            confidence = probs[detected_lang]
            
            return detected_lang, confidence
            
        except Exception as e:
            raise ValueError(f"Error detecting language: {str(e)}")


class FasterWhisperEngine:
    """
    ASR Engine using faster-whisper for improved performance.
    """
    
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        """
        Initialize faster-whisper engine.
        
        Args:
            model_size: Model size ('tiny', 'base', 'small', 'medium', 'large-v2')
            device: Device to use ('cuda' or 'cpu')
            compute_type: Compute type ('int8', 'float16', 'float32')
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("faster-whisper not installed. Run: pip install faster-whisper")
        
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize faster-whisper model."""
        try:
            from faster_whisper import WhisperModel
            
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize faster-whisper model: {str(e)}")
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Tuple[str, float]:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to the audio file
            language: Language code. If None, auto-detect.
            
        Returns:
            Tuple of (transcribed_text, average_confidence)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            segments, info = self.model.transcribe(
                audio_path,
                language=language
            )
            
            # Collect segments
            text_parts = []
            confidences = []
            
            for segment in segments:
                text_parts.append(segment.text)
                confidences.append(segment.avg_logprob)  # Use avg_logprob as confidence proxy
            
            # Combine text
            full_text = " ".join(text_parts).strip()
            
            # Normalize confidences (logprobs are negative, convert to 0-1 range)
            # Typical range is -1.0 to 0.0
            normalized_confidences = [min(1.0, max(0.0, 1.0 + conf)) for conf in confidences]
            avg_confidence = sum(normalized_confidences) / len(normalized_confidences) if normalized_confidences else 0.8
            
            return full_text, avg_confidence
            
        except Exception as e:
            raise ValueError(f"Error transcribing audio: {str(e)}")
