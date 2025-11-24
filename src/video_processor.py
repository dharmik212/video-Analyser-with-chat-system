"""
Video processing and frame extraction
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List
from scenedetect import detect, ContentDetector
import logging

logger = logging.getLogger('video_chat.processor')


class VideoProcessor:
    """Process videos and extract frames intelligently."""
    
    def __init__(self, output_dir: str = "data/frames"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_frames(self, 
                  video_path: str, 
                  frames_per_minute: int = 4,
                  min_frames: int = 4,
                  max_frames: int = None,  # ← None = no limit
                  method: str = "scene") -> List[Path]:
        """
        Extract frames dynamically based on video duration.
        No hard cap - scales with video length.
        
        Args:
            video_path: Path to video file
            frames_per_minute: Core sampling rate
            min_frames: Minimum frames to extract
            max_frames: Maximum frames (None = unlimited)
            method: "scene" or "uniform"
        
        Returns:
            List of frame paths
        """
        video_path = Path(video_path)
        video_name = video_path.stem
        
        # Get video duration
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_minutes = (total_frames / fps) / 60.0
        cap.release()
        
        # Calculate adaptive frame count
        calculated_frames = int(duration_minutes * frames_per_minute)
        
        # Apply constraints
        target_frames = max(min_frames, calculated_frames)
        
        # Apply max only if set
        if max_frames is not None:
            target_frames = min(target_frames, max_frames)
        
        logger.info(f"📹 Extracting frames from {video_name}")
        logger.info(f"   Duration: {duration_minutes:.1f} min ({duration_minutes * 60:.0f}s)")
        logger.info(f"   Target frames: {target_frames} (dynamic: {frames_per_minute}/min)")
        logger.info(f"   Method: {method}")
        
        if method == "scene":
            return self._extract_scene_based(video_path, target_frames)
        else:
            return self._extract_uniform(video_path, target_frames)


    
    def _extract_scene_based(self, video_path: Path, target_frames: int) -> List[Path]:
        """Extract frames based on scene detection - USE ALL SCENES FOUND."""
        
        logger.info("   🔍 Detecting scenes...")
        
        try:
            from scenedetect import detect, ContentDetector
            
            # Detect all scenes
            scenes = detect(str(video_path), ContentDetector(threshold=27.0))
            
            if not scenes or len(scenes) < 2:
                logger.warning("   ⚠️  No scenes detected, using uniform")
                return self._extract_uniform(video_path, target_frames)
            
            logger.info(f"   ✓ Found {len(scenes)} scenes")
            
            # EXTRACT ALL SCENES (no limit!)
            # Use min(target_frames, len(scenes)) only as a fallback for min_frames
            frames_to_extract = len(scenes)
            
            logger.info(f"   📸 Extracting frames from all {frames_to_extract} scenes")
            
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            extracted_frames = []
            
            for scene_idx, scene in enumerate(scenes):
                # Get middle frame of scene
                start_frame = scene[0].get_frames()
                end_frame = scene[1].get_frames()
                middle_frame = (start_frame + end_frame) // 2
                
                # Extract frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Calculate quality score
                quality = self._calculate_quality(frame)
                
                # Save frame
                frame_name = f"{video_path.stem}_scene{scene_idx:02d}_q{int(quality)}.jpg"
                frame_path = self.output_dir / frame_name
                cv2.imwrite(str(frame_path), frame)
                
                extracted_frames.append((frame_path, quality))
            
            cap.release()
            
            # Sort by quality and return ALL (no filtering)
            extracted_frames.sort(key=lambda x: x[1], reverse=True)
            frame_paths = [fp for fp, _ in extracted_frames]
            
            logger.info(f"   ✅ Extracted {len(frame_paths)} frames from scenes")
            
            return frame_paths
            
        except ImportError:
            logger.warning("   ⚠️  PySceneDetect not available, using uniform")
            return self._extract_uniform(video_path, target_frames)
        except Exception as e:
            logger.error(f"   ❌ Scene detection failed: {e}")
            return self._extract_uniform(video_path, target_frames)

    
    def _extract_uniform(self, video_path: Path, max_frames: int) -> List[Path]:
        """Extract evenly spaced frames."""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= max_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        
        video_name = video_path.stem
        frame_paths = []
        
        for idx, frame_num in enumerate(indices):
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if ret:
                    quality = self._calculate_quality(frame)
                    frame_path = self.output_dir / f"{video_name}_uniform{idx:02d}_q{quality:.0f}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(frame_path)
            except Exception as e:
                logger.warning(f"   ⚠️  Error extracting frame {idx}: {e}")
        
        cap.release()
        logger.info(f"   ✅ Extracted {len(frame_paths)} uniform frames")
        return frame_paths
    
    def _calculate_quality(self, frame) -> float:
        """Calculate frame quality score."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            contrast = np.std(gray)
            brightness = np.mean(gray)
            brightness_score = 1 - (abs(brightness - 128) / 128)
            
            quality = (
                min(sharpness / 500, 1.0) * 50 +
                min(contrast / 100, 1.0) * 30 +
                brightness_score * 20
            )
            
            return quality
        except Exception:
            return 50.0  # Default quality score
    
    def get_video_info(self, video_path: str) -> dict:
        """Get video metadata."""
        video_path_obj = Path(video_path)
        
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps == 0 or total_frames == 0:
            cap.release()
            raise ValueError(f"Invalid video file (FPS=0 or no frames): {video_path}")
        
        info = {
            "path": video_path,
            "total_frames": total_frames,
            "fps": fps,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_sec": total_frames / fps if fps > 0 else 0
        }
        
        cap.release()
        return info
