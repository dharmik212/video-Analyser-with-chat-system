# Multi-Video Chat System

An intelligent video analysis platform powered by Qwen2-VL, enabling interactive Q&A across multiple videos with dynamic frame extraction and batch processing.

## Why This Matters

Transform video data into structured, queryable information for rapid analysis and seamless integration with AI and machine learning pipelines.

- Intelligent analysis: Automatically extract key frames using scene detection—no arbitrary limits
- Interactive exploration: Natural language Q&A about video content in real time
- Efficient processing: GPU-accelerated inference with result caching for instant switching
- Scalable architecture: Handles videos of any length with dynamic frame scaling and batch collage processing

## Architecture Overview

```
Pre-Processing Phase (Frame Extraction) → Selection Phase (Lazy Loading) → Chat Phase (Q&A)
├─ Extract frames via scene detection    ├─ Load video on first select  ├─ Answer questions
├─ Generate collage batches              ├─ Run AI analysis once       ├─ Use cached frames
└─ Store in memory (ready for use)       └─ Cache results              └─ Stream responses
```

## Quick Start

### Prerequisites

- Python 3.11+ (required for CUDA support)
- NVIDIA GPU with CUDA 12.0+ (optional but highly recommended)
- 8GB+ RAM (16GB+ recommended for long videos)
- 10GB+ free disk space (for model cache)

### Installation

Install dependencies:

```bash
uv pip install -r requirements.txt
```

Or using pip:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Place videos in the `data/videos/` folder.

### Usage

Run the multi-video chat system:

```bash
python examples/multi_video_chat.py data/videos/
```

## 1. Pre-Processing Phase

Extract frames from all videos simultaneously using scene detection.

**What happens:**

- OpenCV and PySceneDetect identify key moments in each video
- Dynamic frame scaling: Frames are extracted based on video duration (no artificial limits)
- Collage batching: Multiple 3x3 grids for comprehensive coverage
- Results stored in memory, ready for analysis

**Performance Benchmarks (RTX 4060):**

| Video Duration | Frames Extracted | Collages Created | Extraction Time |
|---------------|------------------|------------------|----------------|
| 2 minutes     | 37 scenes        | 5 batches        | ~15s           |
| 30 minutes    | 120+ scenes      | 14+ batches      | ~45s           |

### System Initialization Example

```text
$ python examples/multi_video_chat.py data/videos/

======================================================================
MULTI-VIDEO CHAT SYSTEM
Directory: data/videos
Videos found: 4

Initializing system...
Model loaded successfully!

======================================================================
PRE-PROCESSING ALL VIDEOS
[1/4] Processing: football_match.mp4
✓ Frames extracted: 37
✓ Duration: 120.0s

Pre-processing complete! 4 videos ready.
```

## 2. Selection Phase

Switch between videos and trigger lazy AI analysis on first selection.

**Commands:**

| Command       | Description                             |
|---------------|-----------------------------------------|
| list          | Show all available videos               |
| select <name> | Switch to a video and trigger analysis  |
| info          | Show current video details and frame count |
| back          | Return to video selection menu          |
| history       | Display full conversation history       |
| save          | Export all conversations as JSON        |
| quit          | Exit and cleanup cached frames          |

### First Selection (AI Analysis Triggered)

```text
You: football_match.mp4

Analyzing football_match.mp4 for the first time...

Switched to: football_match.mp4

Initial Analysis (91s):
The video captures a soccer match between Brazil and Morocco in the FIFA World Cup. Key moments include goals, defensive plays, and crowd reactions throughout the 120-minute match.
```

**How it works:**

- Collages loaded into memory (from pre-processing)
- Qwen2-VL processes all collage batches on first selection
- Results cached for instant follow-up questions
- Switching back to the same video uses the cache—no re-analysis required

## 3. Chat Phase

Ask natural language questions about the selected video.

### Interactive Q&A Session

```text
[football_match.mp4] You: Who scored the goal?

A: Based on the video analysis, Brazil's striker Neymar scored a spectacular goal in the 67th minute with a long-range shot from outside the penalty box. Morocco equalized shortly after with a header from a corner kick.
(Answered in 3.4s)

[football_match.mp4] You: What was the final score?

A: The match ended 2-2 with Brazil scoring in the 67th and 89th minutes, while Morocco responded with goals in the 72nd and 85th minutes.
(Answered in 2.8s)

[football_match.mp4] You: back

Returning to video selection
```

**Performance Characteristics:**

- Initial analysis: 40-50s per video (processes all collages)
- Follow-up questions: 3-4s each (cache used)
- Parallel pre-processing: All videos get frames extracted simultaneously

## Configuration

Customize behavior by editing `config.yaml`:

```yaml
model:
  name: "Qwen/Qwen2-VL-2B-Instruct"
  device: "cuda"           # or "cpu"
  dtype: "float16"         # GPU: float16, CPU: float32
  max_new_tokens: 512
  temperature: 0.7

frames:
  method: "scene"          # or "uniform"
  frames_per_minute: 4     # Core sampling rate
  min_frames: 4
  max_frames: null         # null = unlimited (scales with video length)
  collage_batch_size: 9    # 3x3 grid per batch

processing:
  parallel: 4              # Concurrent video processing threads
  cleanup_on_exit: true    # Auto-delete extracted frames
```

## Features

**Core Capabilities**
- Multi-video processing with parallel execution
- Scene-based frame extraction with dynamic scaling
- Interactive chat interface for video content
- GPU acceleration (CUDA support)
- Batch collage processing for long videos
- Result caching for efficient switching between videos
- Conversation history saved and exported as JSON
- Automatic frame cleanup

**Technical Highlights**
- Dynamic frame extraction (no arbitrary limits)
- Batch collages for each segment to maximize context
- Lazy AI analysis on-demand, per-video
- Conversation export for documentation or research

## Output & History Management

### Save Conversations

```text
You: save

Saving all conversations...
football_match_conversation_20251123_214530.json
documentary_conversation_20251123_214530.json

Saved 2 conversation(s)
```

**JSON Format Example:**

```json
{
  "video": "football_match.mp4",
  "duration": 120.0,
  "frames_extracted": 37,
  "timestamp": "2025-11-23T21:45:30",
  "conversation": [
    {
      "question": "Who scored the goal?",
      "answer": "Brazil's striker Neymar scored...",
      "processing_time": 3.4
    }
  ]
}
```

## Technical Stack

| Component            | Technology                          |
|----------------------|-------------------------------------|
| Vision-Language Model| Qwen2-VL-2B-Instruct (Alibaba Cloud)|
| Frame Extraction     | OpenCV, PySceneDetect               |
| Image Processing     | Pillow (PIL)                        |
| Deep Learning        | PyTorch, Transformers               |
| GPU Acceleration     | CUDA 12.1                           |

## Project Structure

```
.
├── examples/
│   └── multi_video_chat.py
├── src/
│   ├── video_processor.py
│   ├── chat_engine.py
│   └── conversation_manager.py
├── config.yaml
├── requirements.txt
└── data/videos/
```

## Performance Benchmarks

**Hardware: RTX 4060 (8GB VRAM)**

2-minute football match:
- Frame extraction: ~15s (37 scenes detected)
- Collages created: 5 (3x3 grids)
- Initial analysis: ~91s
- Follow-up questions: ~3-4s each

30-minute documentary:
- Frame extraction: ~45s (120+ scenes)
- Collages: 14+ batches
- Initial analysis: ~180s
- Follow-up questions: ~40-50s each

## Project Status

Current Version: 1.0.0

**Completed Features**
- Multi-video pre-processing with parallel execution
- Unlimited, dynamic frame extraction
- Scene-based key moment detection
- Batch collage processing (3x3 grids)
- Interactive chat interface
- Lazy AI analysis with first selection
- Result caching for fast switching
- JSON export of conversations
- Automatic frame cleanup

**Planned Features**
- Web UI (Gradio or Streamlit)
- PDF/HTML report generation
- Video upscaling and denoising
- Multi-model support (GPT-4V, Claude Vision)
- Real-time streaming analysis

## Contributing

Contributions are welcome. Please submit pull requests or open issues.

## Credits

- Dharmik Kurlawala — Development & Implementation
- Dmitry Petrov — Project Guidance

## Acknowledgments

- Qwen2-VL by Alibaba Cloud
- PySceneDetect for scene-based frame selection
- Hugging Face Transformers library

## Contact

Dharmik Kurlawala  
kurlawaladharmik@gmail.com

---

Built using Qwen2-VL Vision-Language Model
