"""
Multi-video chat system - Process all videos first, then interactive Q&A on any video
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, setup_logging
from src.video_processor import VideoProcessor
from src.chat_engine import VideoChatEngine
from src.memory_manager import ConversationMemory

logger = setup_logging("INFO")


def find_videos(directory: Path) -> list:
    """Find all video files in directory (case-insensitive)."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
    videos = []
    
    for file in directory.iterdir():
        if file.is_file() and file.suffix.lower() in video_extensions:
            videos.append(file)
    
    return sorted(set(videos))


def preprocess_all_videos(video_files, processor, engine, config):
    """Process all videos and store their data."""
    print(f"\n{'='*70}")
    print(f"🔄 PRE-PROCESSING ALL VIDEOS")
    print(f"{'='*70}\n")
    
    video_data = {}
    
    for idx, video_path in enumerate(video_files, 1):
        print(f"[{idx}/{len(video_files)}] Processing: {video_path.name}")
        
        try:
            # Get video info
            video_info = processor.get_video_info(str(video_path))
            
            # Extract frames
            frames = processor.extract_frames(
            str(video_path),
            frames_per_minute=config['frames']['frames_per_minute'],
            min_frames=config['frames']['min_frames'],
            max_frames=config['frames'].get('max_frames'),  # None if not set
            method=config['frames']['method']
            )
            
            # Load frames and create collage (but don't analyze yet)
            engine.load_video_frames(
            frames,
            batch_size=config['frames'].get('collage_batch_size', 9)
            )
            
            # Create collage cache           
            # Store video data
            video_data[video_path.name] = {
                'path': video_path,
                'info': video_info,
                'frames': frames,
                'collage': engine.collage_cache,
                'images': engine.current_images,  # ← FIX: Store images too
                'analyzed': False,
                'initial_analysis': None
            }
            
            print(f"   ✓ Frames extracted: {len(frames)}")
            print(f"   ✓ Duration: {video_info['duration_sec']:.1f}s\n")
            
        except Exception as e:
            print(f"   ✗ Error: {e}\n")
            logger.error(f"Failed to process {video_path.name}: {e}")
    
    print(f"✅ Pre-processing complete! {len(video_data)} videos ready.\n")
    
    return video_data


def analyze_video_on_demand(video_name, video_data, engine, memory):
    """Analyze a video when user selects it (lazy loading)."""
    
    if video_data[video_name]['analyzed']:
        # Already analyzed, just restore context
        engine.collage_cache = video_data[video_name]['collage']
        engine.current_images = video_data[video_name]['images']
        engine.frame_paths = video_data[video_name]['frames']
        
        # Restore memory context
        memory.clear()
        memory.set_video_context(
            str(video_data[video_name]['path']),
            video_data[video_name]['frames'],
            video_data[video_name]['initial_analysis']
        )
        
        return video_data[video_name]['initial_analysis']
    
    print(f"\n🤖 Analyzing {video_name} for the first time...\n")
    
    # Load the collage and images
    engine.collage_cache = video_data[video_name]['collage']
    engine.current_images = video_data[video_name]['images']
    engine.frame_paths = video_data[video_name]['frames']
    
    # Get initial analysis
    initial_desc, _ = engine.get_initial_description()
    
    # Store it
    video_data[video_name]['analyzed'] = True
    video_data[video_name]['initial_analysis'] = initial_desc
    
    # Set context
    memory.clear()
    memory.set_video_context(
        str(video_data[video_name]['path']),
        video_data[video_name]['frames'],
        initial_desc
    )
    
    return initial_desc

def cleanup_all_frames(video_data):
    """Delete all extracted frames."""
    deleted_count = 0
    for video_name, data in video_data.items():
        for frame_path in data['frames']:
            try:
                if frame_path.exists():
                    frame_path.unlink()
                    deleted_count += 1
            except Exception as e:
                logger.warning(f"Could not delete {frame_path.name}: {e}")
    
    if deleted_count > 0:
        logger.info(f"🗑️  Deleted {deleted_count} frame files")


def interactive_multi_video_chat(video_dir: str, output_dir: str = "data/results/multi_video"):
    """Process all videos, then chat with any video interactively."""
    
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all videos
    video_files = find_videos(video_dir)
    
    if not video_files:
        print(f"❌ No videos found in {video_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"🎬 MULTI-VIDEO CHAT SYSTEM")
    print(f"{'='*70}\n")
    print(f"📁 Directory: {video_dir}")
    print(f"📹 Videos found: {len(video_files)}\n")
    
    # Load config
    config = load_config()
    
    # Initialize components
    print("🔧 Initializing system...")
    processor = VideoProcessor(config['paths']['frames_dir'])
    engine = VideoChatEngine(
        model_name=config['model']['name'],
        device=config['model']['device'],
        dtype=config['model']['dtype']
    )
    memory = ConversationMemory(max_history=10)
    
    # Pre-process all videos
    video_data = preprocess_all_videos(video_files, processor, engine, config)
    
    if not video_data:
        print("❌ No videos could be processed")
        return
    
        # Interactive chat loop
    current_video = None
    session_history = {}  # Track all Q&A per video
    
    print(f"{'='*70}")
    print(f"💬 INTERACTIVE CHAT MODE")
    print(f"{'='*70}\n")
    print("Commands:")
    print("  'list' - Show all available videos")
    print("  'select <name>' or just '<name>' - Switch to a video")
    print("  'back' - Return to video selection (deselect current)")
    print("  'info' - Show current video details")
    print("  'history' - Show conversation history for current video")
    print("  'save' - Save all conversations")
    print("  'exit' or 'quit' - Exit and save\n")
    
    while True:
        # Show current video
        if current_video:
            prompt = f"👤 [{current_video}] You: "
        else:
            prompt = "👤 You: "
        
        user_input = input(prompt).strip()
        
        if not user_input:
            continue
        
        # Commands
        if user_input.lower() in ['quit', 'exit']:
            break
        
        elif user_input.lower() == 'back':
            if current_video:
                print(f"\n⬅️  Returning to video selection\n")
                current_video = None
            else:
                print("\n⚠️  Already at video selection. Use 'list' to see videos.\n")
            continue
        
        elif user_input.lower() == 'list':
            print(f"\n📹 Available videos ({len(video_data)}):")
            for idx, name in enumerate(video_data.keys(), 1):
                status = "✓ analyzed" if video_data[name]['analyzed'] else "○ not analyzed yet"
                qa_count = len(session_history.get(name, []))
                qa_info = f", {qa_count} questions" if qa_count > 0 else ""
                print(f"  {idx}. {name} ({status}{qa_info})")
            print()
            continue
        
        elif user_input.lower().startswith('select '):
            video_name = user_input[7:].strip()
            
            # Fuzzy match
            matches = [v for v in video_data.keys() if video_name.lower() in v.lower()]
            
            if not matches:
                print(f"❌ Video not found: {video_name}\n")
                continue
            
            if len(matches) > 1:
                print(f"⚠️  Multiple matches found:")
                for m in matches:
                    print(f"  - {m}")
                print("Please be more specific.\n")
                continue
            
            current_video = matches[0]
            
            # Analyze if not done yet
            initial_desc = analyze_video_on_demand(current_video, video_data, engine, memory)
            
            print(f"\n✅ Switched to: {current_video}")
            print(f"\n📊 Initial Analysis:")
            print(initial_desc)
            print(f"\n💡 Tip: Type 'back' to return to video selection\n")
            
            # Initialize session history if needed
            if current_video not in session_history:
                session_history[current_video] = []
            
            continue
        
        elif user_input.lower() == 'info':
            if not current_video:
                print("❌ No video selected. Use 'select <name>' or just type video name.\n")
                continue
            
            info = video_data[current_video]['info']
            print(f"\n📹 Video Information:")
            print(f"  Name: {current_video}")
            print(f"  Duration: {info['duration_sec']:.1f}s")
            print(f"  Resolution: {info['width']}x{info['height']}")
            print(f"  FPS: {info['fps']:.1f}")
            print(f"  Frames extracted: {len(video_data[current_video]['frames'])}")
            print(f"  Questions asked: {len(session_history.get(current_video, []))}\n")
            continue
        
        elif user_input.lower() == 'history':
            if not current_video:
                print("❌ No video selected.\n")
                continue
            
            history = session_history.get(current_video, [])
            if not history:
                print(f"📝 No questions asked yet for {current_video}\n")
            else:
                print(f"\n📝 Conversation History for {current_video}:")
                for i, qa in enumerate(history, 1):
                    print(f"\n  Q{i}: {qa['question']}")
                    print(f"  A{i}: {qa['answer'][:100]}...")
                print()
            continue
        
        elif user_input.lower() == 'save':
            # Save all conversations
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            saved_count = 0
            
            for video_name, qa_list in session_history.items():
                if qa_list:
                    result = {
                        "video": video_name,
                        "path": str(video_data[video_name]['path']),
                        "info": video_data[video_name]['info'],
                        "initial_analysis": video_data[video_name]['initial_analysis'],
                        "qa_pairs": qa_list,
                        "total_questions": len(qa_list),
                        "saved_at": datetime.now().isoformat()
                    }
                    
                    output_file = output_dir / f"{Path(video_name).stem}_conversation_{timestamp}.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    
                    print(f"💾 Saved: {output_file.name}")
                    saved_count += 1
            
            if saved_count > 0:
                print(f"\n✅ Saved {saved_count} conversation(s) to {output_dir}\n")
            else:
                print(f"\n⚠️  No conversations to save yet.\n")
            continue
        
        # Check if input is a video name (quick switch)
        if user_input in video_data:
            current_video = user_input
            initial_desc = analyze_video_on_demand(current_video, video_data, engine, memory)
            
            print(f"\n✅ Switched to: {current_video}")
            print(f"\n📊 Initial Analysis:")
            print(initial_desc)
            print(f"\n💡 Tip: Type 'back' to return to video selection\n")
            
            if current_video not in session_history:
                session_history[current_video] = []
            
            continue
        
        # Otherwise, treat as question
        if not current_video:
            print("❌ No video selected. Use 'list' to see videos, then select one.\n")
            print("Quick select: Just type the video name (e.g., test_1.mp4)\n")
            continue
        
        # Ask question about current video
        context = memory.get_context_summary()
        
        answer, time_taken = engine.chat(
            user_input,
            context=context,
            max_tokens=config['model']['max_new_tokens'],
            temperature=config['model']['temperature']
        )
        
        print(f"\n🤖 Assistant: {answer}")
        print(f"   (answered in {time_taken:.2f}s)\n")
        
        # Save to memory and history
        memory.add_exchange(user_input, answer)
        session_history[current_video].append({
            "question": user_input,
            "answer": answer,
            "time_seconds": round(time_taken, 2),
            "timestamp": datetime.now().isoformat()
        })
    
    # Cleanup and save on exit
    print("\n🗑️  Cleaning up frames...")
    cleanup_all_frames(video_data)
    
    print("💾 Saving all conversations...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_count = 0
    
    for video_name, qa_list in session_history.items():
        if qa_list:
            result = {
                "video": video_name,
                "path": str(video_data[video_name]['path']),
                "info": video_data[video_name]['info'],
                "initial_analysis": video_data[video_name]['initial_analysis'],
                "qa_pairs": qa_list,
                "total_questions": len(qa_list),
                "saved_at": datetime.now().isoformat()
            }
            
            output_file = output_dir / f"{Path(video_name).stem}_conversation_{timestamp}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"   ✓ {output_file.name}")
            saved_count += 1
    
    if saved_count > 0:
        print(f"\n✅ Saved {saved_count} conversation(s)")
    else:
        print(f"\n⚠️  No conversations to save")
    
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python multi_video_chat.py <video_directory>")
        print("\nExample: python multi_video_chat.py data/videos/")
        sys.exit(1)
    
    interactive_multi_video_chat(sys.argv[1])
