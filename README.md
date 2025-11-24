# 🎬 Multi-Video Chat System

An intelligent video analysis system powered by **Qwen2-VL** that enables interactive Q&A across multiple videos with dynamic frame extraction and batch processing.

## ✨ Features

### Core Capabilities
- 🎥 **Multi-Video Processing** - Pre-process multiple videos simultaneously
- 🔍 **Intelligent Frame Extraction** - Scene-based detection with dynamic scaling (unlimited frames)
- 💬 **Interactive Chat** - Natural language Q&A about video content
- 🚀 **GPU Acceleration** - CUDA-optimized for fast inference (RTX 4060 tested)
- 📊 **Batch Collage Processing** - Handles videos of any length automatically
- 💾 **Conversation History** - Save and export Q&A sessions as JSON
- 🧹 **Auto Cleanup** - Automatic frame management on exit

### Technical Highlights
- **Dynamic Frame Scaling**: Extracts frames based on video duration (no arbitrary limits)
- **Scene Detection**: Uses PySceneDetect to find key moments
- **Lazy Loading**: AI analysis only when video is selected
- **Result Caching**: Instant switching between analyzed videos
- **Batch Processing**: Multiple 3x3 collages for comprehensive coverage

## 🏗️ Architecture

video-chat-system/
├── src/ # Core modules
│ ├── chat_engine.py # Qwen2-VL chat interface
│ ├── video_processor.py # Frame extraction & scene detection
│ ├── memory_manager.py # Conversation context management
│ └── utils.py # Configuration & logging
├── examples/ # Usage examples
│ └── multi_video_chat.py # Main interactive system
├── data/
│ ├── videos/ # Input videos
│ ├── frames/ # Extracted frames (temporary)
│ └── results/ # JSON conversation exports
├── logs/ # Conversation logs
├── config.yaml # System configuration
└── requirements.txt # Dependencies


## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** (required for CUDA support)
- **NVIDIA GPU** with CUDA 12.0+ (optional but highly recommended)
- **8GB+ RAM** (16GB+ recommended for long videos)
- **10GB+ free disk space** (for model cache)

### Installation

1. **Clone the repository:**
git clone <repository-url>
cd video-chat-system

2. **Create virtual environment:**
python -m venv venv

Windows:
.\venv\Scripts\activate

Linux/Mac:
source venv/bin/activate

3. **Install PyTorch with CUDA** (for GPU acceleration):
CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

CPU only (slower)
pip install torch torchvision torchaudio

4. **Install dependencies:**
pip install -r requirements.txt

5. **Place videos in `data/videos/` folder**

### Usage

**Run the multi-video chat system:**
python examples/multi_video_chat.py data/videos/

### Commands

| Command | Description |
|---------|-------------|
| `list` | Show all available videos |
| `select <name>` or `<name>` | Switch to a video |
| `back` | Return to video selection |
| `info` | Show current video details |
| `history` | Show conversation history |
| `save` | Save all conversations |
| `quit` | Exit and cleanup |

### Example Session

$ python examples/multi_video_chat.py data/videos/

======================================================================
🎬 MULTI-VIDEO CHAT SYSTEM
📁 Directory: data/videos
📹 Videos found: 4

🔧 Initializing system...
✅ Model loaded successfully!

======================================================================
🔄 PRE-PROCESSING ALL VIDEOS
[1/4] Processing: football_match.mp4
✓ Frames extracted: 37
✓ Duration: 120.0s

✅ Pre-processing complete! 4 videos ready.

======================================================================
💬 INTERACTIVE CHAT MODE
👤 You: football_match.mp4

🤖 Analyzing football_match.mp4 for the first time...

✅ Switched to: football_match.mp4

📊 Initial Analysis:
The video captures a soccer match between Brazil and Morocco...

👤 [football_match.mp4] You: Who scored the goal?

🤖 Assistant: Based on the video analysis, Brazil scored...
(answered in 3.4s)

👤 [football_match.mp4] You: back

⬅️ Returning to video selection

👤 You: quit

🗑️ Cleaning up frames...
💾 Saving all conversations...
✓ football_match_conversation_20251123_214530.json

✅ Saved 1 conversation(s)

👋 Goodbye!

## ⚙️ Configuration

Edit `config.yaml` to customize behavior:

model:
name: "Qwen/Qwen2-VL-2B-Instruct"
device: "cuda" # or "cpu"
dtype: "float16" # GPU: float16, CPU: float32
max_new_tokens: 512
temperature: 0.7

frames:
method: "scene" # or "uniform"
frames_per_minute: 4 # Core sampling rate
min_frames: 4
max_frames: null # null = unlimited (scales with video length)
collage_batch_size: 9 # 3x3 grid per batch


## 📊 Performance

**Hardware: RTX 4060 (8GB VRAM)**

**2-minute football match (37 scenes detected):**
- Frame extraction: ~15s
- Frames extracted: 37 (all scenes)
- Collages created: 5 (3x3 grids)
- Initial analysis: ~91s
- Follow-up questions: ~3-4s each

**30-minute video (120+ scenes):**
- Frames extracted: 120+
- Collages: 14+ batches
- Questions: ~40-50s each (processes all collages)

## 🔬 How It Works

### 1. Pre-Processing Phase (No AI, Fast)
for video in videos:
frames = extract_frames(video) # OpenCV + PySceneDetect
collages = create_collages(frames) # PIL
store_in_memory(frames, collages) # Ready for analysis

### 2. Selection Phase (Lazy Loading)
if user_selects(video):
if not already_analyzed(video):
analysis = run_ai_analysis(video) # First time only
cache_result(analysis)
show_analysis()

### 3. Chat Phase (Fast Q&A)
answer = chat_with_video(
question=user_question,
collages=cached_collages # Already in memory
)

## 🛠️ Technical Stack

- **Vision-Language Model**: Qwen2-VL-2B-Instruct (Alibaba Cloud)
- **Frame Extraction**: OpenCV + PySceneDetect
- **Image Processing**: Pillow (PIL)
- **Deep Learning**: PyTorch + Transformers
- **GPU Acceleration**: CUDA 12.1

## 📝 Project Status

**Current Version**: 1.0.0

**Completed Features:**
- ✅ Multi-video pre-processing
- ✅ Dynamic frame extraction (unlimited)
- ✅ Scene-based detection
- ✅ Batch collage processing
- ✅ Interactive chat interface
- ✅ Lazy AI analysis
- ✅ Result caching
- ✅ Conversation export
- ✅ Auto frame cleanup

**Planned Features:**
- 🔄 Web UI (Gradio interface)
- 🔄 PDF/HTML report generation
- 🔄 Video preprocessing (upscaling, denoising)
- 🔄 Multi-model support (GPT-4V, Claude Vision)

## 🤝 Contributing

Contributions welcome! Please feel free to submit pull requests.

## 📄 License

[Your License Here - e.g., MIT]

## 👥 Contributors

- **Dharmik** - Development & Implementation
- **Dmitry Petrov** - Project Guidance

## 🙏 Acknowledgments

- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) by Alibaba Cloud
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) for intelligent frame selection
- [Hugging Face Transformers](https://github.com/huggingface/transformers) library

## 📧 Contact

Dharmik Kurlawala
kurlawaladharmik@gmail.com

---

**Built using Qwen2-VL**
