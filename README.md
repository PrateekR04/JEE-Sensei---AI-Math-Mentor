# Math Mentor AI - Day 1 Foundation

A production-ready multimodal AI Math Mentor application with OCR and ASR capabilities.

## 🚀 Features

### Day 1 Implementation
- **Multimodal Input Support**
  - 📝 Text input
  - 🖼️ Image upload with OCR (PaddleOCR)
  - 🎤 Audio upload with ASR (Whisper)

- **OCR Pipeline**
  - PaddleOCR integration
  - Text extraction from images (JPG/PNG)
  - Confidence score calculation
  - Editable preview box

- **ASR Pipeline**
  - OpenAI Whisper integration
  - Audio transcription (WAV/MP3/M4A/OGG)
  - Confidence score calculation
  - Editable transcript preview

- **User Experience**
  - Clean Streamlit UI
  - Input mode selector
  - Editable text confirmation
  - Confidence score display
  - Real-time processing feedback

## 📁 Project Structure

```
math_mentor_ai/
├── app.py                    # Main Streamlit application
├── ocr/
│   ├── __init__.py
│   └── ocr_engine.py        # PaddleOCR integration
├── asr/
│   ├── __init__.py
│   └── whisper_engine.py    # Whisper ASR integration
├── utils/
│   ├── __init__.py
│   └── confidence.py        # Confidence calculation utilities
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd math_mentor_ai
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   **Note:** The first run will download the Whisper model (~140MB for base model) and PaddleOCR models automatically.

## 🎯 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the Application

1. **Select Input Mode** (from sidebar):
   - **Text**: Type your math problem directly
   - **Image (OCR)**: Upload an image containing a math problem
   - **Audio (ASR)**: Upload an audio recording of your question

2. **Process Input**:
   - For images: Click "Extract Text (OCR)"
   - For audio: Click "Transcribe Audio (ASR)"
   - For text: Click "Submit"

3. **Review & Edit**:
   - Review the extracted/transcribed text
   - Check the confidence score
   - Edit the text if needed

4. **Confirm**:
   - Click "Confirm & Submit" to finalize your input
   - The input is now ready for backend processing

## 🔧 Configuration

### GPU Support

To enable GPU acceleration (requires CUDA):

1. Edit `requirements.txt`:
   ```
   # Comment out CPU version
   # paddlepaddle==2.6.0
   
   # Uncomment GPU version
   paddlepaddle-gpu==2.6.0
   ```

2. Update OCR initialization in `app.py`:
   ```python
   return OCREngine(use_gpu=True, lang='en')
   ```

### Whisper Model Size

To use a different Whisper model (trade-off between speed and accuracy):

Edit `app.py`:
```python
# Options: tiny, base, small, medium, large
return WhisperEngine(model_size="small", device="cpu")
```

**Model Sizes:**
- `tiny`: Fastest, least accurate (~39M parameters)
- `base`: Good balance (~74M parameters) - **Default**
- `small`: Better accuracy (~244M parameters)
- `medium`: High accuracy (~769M parameters)
- `large`: Best accuracy (~1550M parameters)

## 📦 Dependencies

### Core
- `streamlit` - Web UI framework
- `numpy` - Numerical operations
- `pillow` - Image processing

### OCR
- `paddleocr` - OCR engine
- `paddlepaddle` - Deep learning framework

### ASR
- `openai-whisper` - Speech recognition
- `faster-whisper` - Optimized Whisper implementation

### Audio Processing
- `soundfile` - Audio I/O
- `librosa` - Audio analysis

## 🏗️ Architecture

### Modular Design

- **`app.py`**: Main Streamlit application with UI logic
- **`ocr/ocr_engine.py`**: OCR functionality encapsulated in `OCREngine` class
- **`asr/whisper_engine.py`**: ASR functionality with `WhisperEngine` and `FasterWhisperEngine` classes
- **`utils/confidence.py`**: Reusable confidence calculation utilities

### Error Handling

- File validation
- Format checking
- Graceful fallbacks
- User-friendly error messages

### Performance Optimization

- Model caching with `@st.cache_resource`
- Temporary file cleanup
- CPU/GPU flexibility

## 🧪 Testing

### Manual Testing

1. **Text Input**: Enter a simple math problem
2. **Image OCR**: Upload a clear image with printed/handwritten math
3. **Audio ASR**: Upload a clear audio recording

### Sample Test Cases

- **OCR**: Upload an image of a math equation from a textbook
- **ASR**: Record "What is 2 plus 2?" and upload
- **Text**: Type "Solve x squared equals 16"

## 🚧 Future Enhancements (Post Day-1)

- [ ] LLM integration for math problem solving
- [ ] Step-by-step solution generation
- [ ] Math notation rendering (LaTeX)
- [ ] Solution history
- [ ] User authentication
- [ ] Database integration
- [ ] API endpoints
- [ ] Batch processing
- [ ] Multi-language support

## 📝 Notes

### Production Considerations

- **Security**: No secrets in code, environment variables for API keys
- **Scalability**: Modular architecture allows easy scaling
- **Maintainability**: Clean code with docstrings and type hints
- **Extensibility**: Easy to add new input modes or processing pipelines

### Known Limitations

- First run downloads models (one-time setup)
- Large audio files may take time to process
- OCR accuracy depends on image quality
- Whisper requires significant memory for larger models

## 📄 License

This is a Day-1 foundation project for educational purposes.

## 🤝 Contributing

This is a foundational implementation. Future contributions welcome for:
- Additional input modes
- Improved error handling
- Performance optimizations
- UI/UX enhancements

---

**Built with ❤️ using Streamlit, EasyOCR, and Whisper**
