"""
Conversational video understanding engine with Qwen2-VL
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from pathlib import Path
from typing import List
import logging
import time

logger = logging.getLogger('video_chat.engine')


class VideoChatEngine:
    """Interactive chat engine using Qwen2-VL."""
    
    def __init__(self, 
             model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
             device: str = "auto",
             dtype: str = "float32"):
        """Initialize chat engine with Qwen2-VL."""
        self.model_name = model_name
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.dtype = torch.float32 if dtype == "float32" else torch.float16
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🤖 VIDEO CHAT ENGINE (Qwen2-VL)")
        logger.info(f"{'='*70}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Dtype: {dtype}")
        logger.info(f"{'='*70}\n")
        
        self._load_model()
        self.current_images = None
        self.collage_cache = None
        self.frame_paths = None  # Track frame paths for cleanup

    
    def _load_model(self):
        """Load Qwen2-VL model."""
        logger.info("🔧 Loading model...")
        
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map=self.device
            )
            
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            self.model.eval()
            
            logger.info("✅ Model loaded successfully!")
            logger.info(f"   Parameters: ~2B")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def load_video_frames(self, frame_paths: List[Path], batch_size: int = 9):
        """
        Load frames into memory with batch collage support.
        For large frame sets, creates multiple collages.
        """
        logger.info(f"📸 Loading {len(frame_paths)} frames into memory...")
        
        # Store frame paths for later cleanup
        self.frame_paths = frame_paths
        
        self.current_images = [
            Image.open(str(fp)).convert('RGB') 
            for fp in frame_paths
        ]
        
        # Create collage(s)
        if len(self.current_images) <= batch_size:
            # Single collage
            logger.info("   Creating single collage from frames...")
            self.collage_cache = self._create_frame_collage(
                self.current_images, 
                max_frames=batch_size
            )
            self.collage_batches = [self.collage_cache]  # Store as list for consistency
        else:
            # Multiple collages for large frame sets
            logger.info(f"   Creating multiple collages ({len(self.current_images)} frames / {batch_size} per batch)...")
            self.collage_batches = []
            
            for i in range(0, len(self.current_images), batch_size):
                batch = self.current_images[i:i+batch_size]
                collage = self._create_frame_collage(batch, max_frames=batch_size)
                self.collage_batches.append(collage)
            
            logger.info(f"   Created {len(self.collage_batches)} collages")
            
            # Set first collage as primary (for compatibility)
            self.collage_cache = self.collage_batches[0]
        
        logger.info(f"✅ Frames loaded and processed")
    
    def _create_frame_collage(self, images: List[Image.Image], max_frames: int = 9) -> Image.Image:
        """Create grid collage (now supports up to 9 frames in 3x3)."""
        images = images[:max_frames]
        
        n = len(images)
        if n == 1:
            return images[0]
        
        # Determine grid size based on frame count
        if n <= 4:
            cols, rows = 2, 2  # 2x2 for 4 or fewer frames
        elif n <= 9:
            cols, rows = 3, 3  # 3x3 for 5-9 frames
        else:
            cols, rows = 3, 4  # 3x4 for 10-12 frames
        
        target_size = (512, 512)
        resized = [img.resize(target_size, Image.LANCZOS) for img in images]
        
        # Pad with black if needed to fill grid
        while len(resized) < (cols * rows):
            resized.append(Image.new('RGB', target_size, (0, 0, 0)))
        
        collage_width = cols * target_size[0]
        collage_height = rows * target_size[1]
        collage = Image.new('RGB', (collage_width, collage_height), (0, 0, 0))
        
        for idx, img in enumerate(resized[:cols*rows]):
            row = idx // cols
            col = idx % cols
            x = col * target_size[0]
            y = row * target_size[1]
            collage.paste(img, (x, y))
        
        logger.info(f"   Created {cols}x{rows} collage ({n} frames)")
        
        return collage

    
    def chat(self, 
         question: str,
         context: str = "",
         max_tokens: int = 512,
         temperature: float = 0.7) -> tuple:
        """
        Chat with support for multiple collages (batched processing).
        """
        
        if not hasattr(self, 'current_images') or self.current_images is None:
            return "❌ No video loaded. Please load a video first.", 0.0
        
        logger.info(f"💬 Question: {question}")
        
        start_time = time.time()
        
        # Check if we have multiple collages
        if hasattr(self, 'collage_batches') and len(self.collage_batches) > 1:
            logger.info(f"   Processing {len(self.collage_batches)} collage batches...")
            
            # Process each collage and aggregate answers
            batch_answers = []
            
            for idx, collage in enumerate(self.collage_batches):
                logger.info(f"   Processing batch {idx+1}/{len(self.collage_batches)}...")
                
                # Build prompt with context
                if context:
                    full_question = f"Context: {context}\n\nQuestion: {question}\n\nNote: This is part {idx+1} of {len(self.collage_batches)} from the video."
                else:
                    full_question = f"{question}\n\nNote: This is part {idx+1} of {len(self.collage_batches)} from the video."
                
                # Create message with collage
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": collage},
                            {"type": "text", "text": full_question}
                        ]
                    }
                ]
                
                # Process
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(self.device)
                
                # Generate
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature
                    )
                
                generated_ids = [
                    output_ids[len(input_ids):] 
                    for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                batch_answer = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                batch_answers.append(batch_answer)
            
            # Aggregate answers
            if len(batch_answers) == 1:
                answer = batch_answers[0]
            else:
                # Synthesize comprehensive answer
                synthesis_prompt = f"""Based on analysis of {len(batch_answers)} segments of the video, synthesize a comprehensive answer to: "{question}"

    Segment answers:
    {chr(10).join([f"Segment {i+1}: {ans}" for i, ans in enumerate(batch_answers)])}

    Provide a unified, coherent answer:"""
                
                # Quick synthesis (could use model again, but simpler for now)
                answer = f"Comprehensive analysis across {len(batch_answers)} segments:\n\n"
                answer += "\n\n".join([f"Part {i+1}: {ans}" for i, ans in enumerate(batch_answers)])
            
        else:
            # Single collage - original logic
            logger.info("   Using processed image")
            
            if context:
                full_question = f"Context: {context}\n\nQuestion: {question}"
            else:
                full_question = question
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": self.collage_cache},
                        {"type": "text", "text": full_question}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature
                )
            
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            answer = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        
        time_taken = time.time() - start_time
        
        logger.info(f"🤖 Answer: {answer}")
        logger.info(f"⏱️  Time: {time_taken:.2f}s")
        
        return answer, time_taken

    
    def get_initial_description(self) -> tuple:
        """Get initial video analysis."""
        question = """Analyze this video comprehensively. Describe:
            1. What is happening (main action/scene)
            2. Who or what is in the frame
            3. The setting/environment
            4. Overall quality and clarity
            5. Any notable details or moments"""
        
        return self.chat(question, max_tokens=600)
    
    def clear_frames(self):
        """Clear loaded frames from memory and delete files."""
        # Clean up frame files
        if hasattr(self, 'frame_paths') and self.frame_paths:
            for fp in self.frame_paths:
                try:
                    if fp.exists():
                        fp.unlink()
                except Exception as e:
                    logger.warning(f"   Could not delete {fp.name}: {e}")
            logger.info("🗑️  Frame files deleted")
        
        # Clear memory
        self.current_images = None
        self.collage_cache = None
        self.frame_paths = None
        logger.info("🗑️  Frames cleared from memory")

