"""
Video Compression Service API Server

This module provides a FastAPI server for video compression services.
It handles video upload, compression, and storage operations.
"""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger

from video_preprocessor import pre_processing
from scene_detector import scene_detection
from encoder import ai_encoding, load_encoding_resources
from iterative_encoder import iterative_encode
from vmaf_calculator import scene_vmaf_calculation
from validator_merger import validation_and_merging
from vidaio_subnet_core.utilities import storage_client, download_video
from vidaio_subnet_core import CONFIG
from utils.video_utils import get_video_duration, get_video_codec


# ============================================================================
# Configuration Variables
# ============================================================================

# VMAF threshold to quality level mapping (configurable for miner flow)
VMAF_THRESHOLD_HIGH = 93.0
VMAF_THRESHOLD_MEDIUM = 89.0
VMAF_THRESHOLD_LOW = 85.0

# ============================================================================
# FastAPI Application Setup
# ============================================================================

app = FastAPI(title="Video Compression Service", version="1.1.0")


# ============================================================================
# Data Models
# ============================================================================

class CompressPayload(BaseModel):
    """Payload for video compression requests."""
    payload_url: str
    vmaf_threshold: float
    target_codec: str = 'av1'  # Target codec: av1, hevc, h264, vp9
    codec_mode: str = 'CRF'  # Codec mode: CRF (Constant Rate Factor), CBR (Constant Bitrate), VBR (Variable Bitrate)
    target_bitrate: float = 10.0  # Target bitrate in Mbps (for CBR/VBR modes)
    target_quality: str = 'Medium'  # High, Medium, Low (legacy, derived from VMAF) (legacy, derived from VMAF)
    max_duration: int = 3600  # Maximum allowed video duration in seconds
    output_dir: str = './output'  # Output directory for final files


class TestCompressPayload(BaseModel):
    """Payload for test compression requests."""
    video_path: str


# ============================================================================
# Helper Functions
# ============================================================================

def create_lightweight_metadata(input_file: str, target_quality: str, target_codec: str, max_duration: int = 3600) -> Optional[dict]:
    """
    Create video metadata without preprocessing (for already-compressed videos).

    This is a lightweight alternative to pre_processing() that only extracts
    metadata without re-encoding. Perfect for miner chunks that are already compressed.

    Args:
        input_file: Path to input video file
        target_quality: Target quality level ('High', 'Medium', 'Low')
        target_codec: Target codec for encoding
        max_duration: Maximum allowed duration

    Returns:
        dict: Video metadata or None if validation fails
    """
    print(f"\n⚡ === Part 1: Pre-processing (SKIPPED - Lightweight Metadata) ===")
    print(f"   📏 Extracting metadata from already-compressed video")

    # Map quality to VMAF using configurable thresholds
    quality_vmaf_mapping = {
        'High': VMAF_THRESHOLD_HIGH,
        'Medium': VMAF_THRESHOLD_MEDIUM,
        'Low': VMAF_THRESHOLD_LOW
    }
    target_vmaf = quality_vmaf_mapping.get(target_quality, VMAF_THRESHOLD_MEDIUM)

    # Get video duration
    duration = get_video_duration(input_file)
    if duration is None:
        print("   ❌ Could not determine video duration")
        return None

    if duration > max_duration:
        print(f"   ❌ Video duration {duration}s exceeds limit of {max_duration}s")
        return None

    # Get video codec
    original_codec = get_video_codec(input_file)
    if not original_codec:
        print("   ❌ Could not determine video codec")
        return None

    print(f"   ✅ Duration: {duration:.1f}s")
    print(f"   ✅ Codec: {original_codec}")
    print(f"   🎯 Target: {target_quality} (VMAF: {target_vmaf})")
    print(f"   🎥 Target codec: {target_codec}")

    # Return lightweight metadata (same format as pre_processing)
    return {
        'path': input_file,
        'codec': original_codec,
        'original_codec': original_codec,
        'duration': duration,
        'was_reencoded': False,
        'encoding_time': 0.0,
        'target_vmaf': target_vmaf,
        'target_quality': target_quality,
        'target_codec': target_codec,
        'processing_info': {
            'lossless_conversion': False,
            'skipped_preprocessing': True
        }
    }


# ============================================================================
# Codec Mapping
# ============================================================================

def map_codec_name(target_codec: str, prefer_gpu: bool = True) -> str:
    """
    Map user-facing codec names (from ffprobe format) to ffmpeg encoder names.

    The protocol uses standard codec names (av1, hevc, h264, vp9) which match
    ffprobe output, but ffmpeg requires specific encoder names.

    Args:
        target_codec: Codec name from protocol (av1, hevc, h264, vp9)
        prefer_gpu: Whether to prefer GPU encoders (NVENC) when available

    Returns:
        FFmpeg encoder name (e.g., av1_nvenc, libx265, libx264, libvpx-vp9)

    Examples:
        map_codec_name('av1', prefer_gpu=True) → 'av1_nvenc'
        map_codec_name('av1', prefer_gpu=False) → 'libsvtav1'
        map_codec_name('hevc', prefer_gpu=True) → 'hevc_nvenc'
        map_codec_name('h264', prefer_gpu=False) → 'libx264'
    """
    codec_map = {
        'av1': 'av1_nvenc' if prefer_gpu else 'libsvtav1',
        'hevc': 'hevc_nvenc' if prefer_gpu else 'libx265',
        'h264': 'h264_nvenc' if prefer_gpu else 'libx264',
        'vp9': 'libvpx_vp9',  # No NVENC encoder for VP9
    }

    # Normalize to lowercase and get mapped codec
    normalized_codec = target_codec.lower().strip()
    ffmpeg_codec = codec_map.get(normalized_codec)

    if not ffmpeg_codec:
        logger.warning(f"Unknown codec '{target_codec}', defaulting to av1_nvenc")
        return 'av1_nvenc'

    logger.info(f"Mapped codec '{target_codec}' → '{ffmpeg_codec}' (GPU={prefer_gpu})")
    return ffmpeg_codec


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/compress-video")
async def compress_video(video: CompressPayload):
    """
    Compress a video from a URL payload.

    Args:
        video: Compression request payload

    Returns:
        dict: Compression results with uploaded video URL
    """
    request_start = time.time()
    print(f"video url: {video.payload_url}")
    print(f"vmaf threshold: {video.vmaf_threshold}")
    print(f"target codec: {video.target_codec}")
    print(f"codec mode: {video.codec_mode}")
    print(f"target bitrate: {video.target_bitrate} Mbps")

    # Download with retry — transient S3 errors sometimes resolve on retry
    input_path = None
    last_dl_err = None
    for attempt in range(3):
        try:
            input_path = await download_video(video.payload_url)
            break
        except Exception as dl_err:
            last_dl_err = dl_err
            elapsed = time.time() - request_start
            print(f"Download attempt {attempt+1}/3 failed after {elapsed:.1f}s: {dl_err}")
            if "404" in str(dl_err) or "403" in str(dl_err) or elapsed > 30:
                break
            await asyncio.sleep(2)
    if input_path is None:
        print(f"Download failed permanently: {last_dl_err}")
        return {"uploaded_video_url": None, "status": "download_failed",
                "detail": f"Could not download source video: {last_dl_err}"}
    download_time = time.time() - request_start
    print(f"Download took {download_time:.1f}s, {135 - download_time:.0f}s remaining for encode+upload")
    input_file = Path(input_path)
    vmaf_threshold = video.vmaf_threshold

    # Map VMAF threshold to target quality using configurable thresholds
    if vmaf_threshold == VMAF_THRESHOLD_LOW:
        target_quality = 'Low'
    elif vmaf_threshold == VMAF_THRESHOLD_MEDIUM:
        target_quality = 'Medium'
    elif vmaf_threshold == VMAF_THRESHOLD_HIGH:
        target_quality = 'High'
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid VMAF threshold. Expected {VMAF_THRESHOLD_LOW}, {VMAF_THRESHOLD_MEDIUM}, or {VMAF_THRESHOLD_HIGH}, got {vmaf_threshold}"
        )

    # AV1: GPU (NVENC)
    # HEVC: CPU (libx265) — better compression efficiency per VMAF point than hevc_nvenc
    use_gpu = video.target_codec.lower() != 'hevc'
    ffmpeg_codec = map_codec_name(video.target_codec, prefer_gpu=use_gpu)

    # Validate input file
    if not input_file.is_file():
        raise HTTPException(status_code=400, detail="Input video file does not exist.")

    # Create output directory
    output_dir = Path(video.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate real time remaining (accounts for download time and queue wait)
    elapsed_so_far = time.time() - request_start
    remaining_for_pipeline = max(30, 135 - elapsed_so_far - 10)  # 10s for upload
    print(f"Time budget for pipeline: {remaining_for_pipeline:.0f}s (elapsed so far: {elapsed_so_far:.1f}s)")

    # Perform video compression
    try:
        compressed_video_path = video_compressor(
            input_file=str(input_file),
            target_quality=target_quality,
            target_codec=ffmpeg_codec,
            codec_mode=video.codec_mode,
            target_bitrate=video.target_bitrate,
            max_duration=video.max_duration,
            output_dir=str(output_dir),
            skip_scene_detection=True,
            skip_preprocessing=True,
            time_budget=remaining_for_pipeline,
        )
        print(f"compressed_video_path: {compressed_video_path}")

        if compressed_video_path and Path(compressed_video_path).exists():
            # Upload compressed video to storage
            try:
                compressed_video_name = os.path.basename(compressed_video_path)
                object_name: str = compressed_video_name
                
                # Upload file
                await storage_client.upload_file(object_name, compressed_video_path)
                print(f"object_name: {object_name}")
                print("Video uploaded successfully.")
                
                # Clean up local file
                if os.path.exists(compressed_video_path):
                    os.remove(compressed_video_path)
                    print(f"{compressed_video_path} has been deleted.")
                else:
                    print(f"{compressed_video_path} does not exist.")
                
                # Get sharing link
                sharing_link: Optional[str] = await storage_client.get_presigned_url(object_name)
                print(f"sharing_link: {sharing_link}")
                
                if not sharing_link:
                    print("Upload failed")
                    return {"uploaded_video_url": None}
                
                return {
                    "uploaded_video_url": sharing_link,
                    "status": "success",
                    "compressed_video_path": str(compressed_video_path)
                }
            except Exception as upload_error:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to upload compressed video: {str(upload_error)}"
                )
        else:
            raise HTTPException(status_code=500, detail="Video compression failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video compression error: {str(e)}")


@app.post("/test-compress")
async def test_compress_video(test_payload: TestCompressPayload):
    """
    Test endpoint for video compression using local video path.
    
    Args:
        test_payload: Test compression request payload
        
    Returns:
        dict: Test compression results
    """
    video_path = Path(test_payload.video_path)
    
    # Validate input file
    if not video_path.is_file():
        raise HTTPException(
            status_code=400, 
            detail=f"Video file does not exist: {video_path}"
        )
    
    try:
        # Perform test compression
        compressed_video_path = test_video_compression(str(video_path))
        
        if compressed_video_path and Path(compressed_video_path).exists():
            return {
                "status": "success",
                "message": "Video compression test completed successfully",
                "input_path": str(video_path),
                "output_path": compressed_video_path,
                "output_size_mb": round(
                    Path(compressed_video_path).stat().st_size / (1024 * 1024), 2
                )
            }
        else:
            raise HTTPException(status_code=500, detail="Video compression test failed")
            
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Video compression test error: {str(e)}"
        )


# ============================================================================
# Core Video Compression Functions
# ============================================================================

def video_compressor(
    input_file: str,
    target_quality: str = 'Medium',
    target_codec: str = 'av1_nvenc',
    codec_mode: str = 'CRF',
    target_bitrate: float = 10.0,
    max_duration: int = 3600,
    output_dir: str = './output',
    skip_scene_detection: bool = True,
    skip_preprocessing: bool = True,
    time_budget: float = 120.0,
) -> Optional[str]:
    """
    Main video compression pipeline orchestrator.

    Args:
        input_file: Path to input video file
        target_quality: Target quality level ('High', 'Medium', 'Low')
        target_codec: FFmpeg encoder name (e.g., 'av1_nvenc', 'hevc_nvenc', 'libx264')
        codec_mode: Encoding mode - 'CRF' (Constant Rate Factor), 'CBR' (Constant Bitrate), 'VBR' (Variable Bitrate)
        target_bitrate: Target bitrate in Mbps (used for CBR/VBR modes)
        max_duration: Maximum allowed video duration in seconds
        output_dir: Output directory for final files
        skip_scene_detection: If True, treats entire video as single scene (default: True for miner chunks)
        skip_preprocessing: If True, skips lossless re-encoding (default: True for already-compressed miner chunks)

    Returns:
        str: Path to compressed video file, or None if failed
    """
    # Record pipeline start time
    pipeline_start_time = time.time()
    
    # Get current directory and setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = _load_configuration(current_dir)
    config['directories']['output_dir'] = str(output_dir_path)
    if 'video_processing' not in config:
        config['video_processing'] = {}
    config['video_processing']['target_quality'] = target_quality
    config['video_processing']['target_codec'] = target_codec
    config['video_processing']['codec_mode'] = codec_mode
    config['video_processing']['target_bitrate'] = target_bitrate

    # Create temp directory
    temp_dir = Path(config['directories']['temp_dir'])
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Display pipeline information
    _display_pipeline_info(input_file, target_quality, max_duration, output_dir)

    # PART 1: Pre-processing (optional - can be skipped for already-compressed videos)
    if skip_preprocessing:
        # Skip full preprocessing - use lightweight metadata extraction
        part1_result = create_lightweight_metadata(input_file, target_quality, target_codec, max_duration)
        if not part1_result:
            print("❌ Part 1 failed. Pipeline terminated.")
            return False
        part1_time = time.time() - pipeline_start_time
        print(f"   ⏱️ Metadata extraction: {part1_time:.2f}s")
    else:
        # Run full preprocessing (checks for lossless codecs and re-encodes if needed)
        part1_result = _execute_preprocessing(input_file, target_quality, max_duration, output_dir_path)
        if not part1_result:
            print("❌ Part 1 failed. Pipeline terminated.")
            return False
        part1_time = time.time() - pipeline_start_time
        _display_preprocessing_results(part1_result, part1_time)

    # PART 2: Scene Detection (optional - can be skipped for pre-chunked videos)
    part2_start_time = time.time()

    if skip_scene_detection:
        # Skip scene detection - treat entire video as single scene
        print(f"\n⚡ === Part 2: Scene Detection (SKIPPED) ===")
        print(f"   📏 Treating entire video as single scene (pre-chunked input)")

        scenes_metadata = [{
            'path': part1_result['path'],
            'scene_number': 1,
            'start_time': 0.0,
            'end_time': part1_result['duration'],
            'duration': part1_result['duration'],
            'original_video_metadata': part1_result
        }]
        part2_time = time.time() - part2_start_time
        print(f"   ⏱️ Scene setup: {part2_time:.2f}s")
        print(f"   ✅ 1 scene created (0.0s - {part1_result['duration']:.1f}s)")
    else:
        # Run normal scene detection
        scenes_metadata = scene_detection(part1_result)
        if not scenes_metadata:
            print("❌ Part 2 failed. Pipeline terminated.")
            return False

        part2_time = time.time() - part2_start_time
        _display_scene_detection_results(scenes_metadata, part2_time)
    
    # PART 3: AI Encoding — pass remaining time budget
    elapsed_pipeline = time.time() - pipeline_start_time
    encode_budget = max(30, time_budget - elapsed_pipeline)
    part3_result = _execute_ai_encoding(scenes_metadata, config, target_quality,
                                        encode_budget=encode_budget)
    if not part3_result:
        print("❌ Part 3 failed completely. Pipeline terminated.")
        return False
    
    part3_time = part3_result['processing_time']
    encoded_scenes_data = part3_result['encoded_scenes_data']
    successful_encodings = part3_result['successful_encodings']

    # PART 4: Validation and Merging
    part4_result = _execute_validation_and_merging(
        part1_result, encoded_scenes_data, config
    )
    if not part4_result:
        print("❌ Part 4 failed. Could not create final video.")
        return False
    
    part4_time = part4_result['processing_time']
    final_video_path = part4_result['final_video_path']
    final_vmaf = part4_result['final_vmaf']
    comprehensive_report = part4_result['comprehensive_report']
    
    _display_validation_results(part4_result)

    # Pipeline completion summary
    total_pipeline_time = time.time() - pipeline_start_time
    _display_pipeline_summary(
        input_file, final_video_path, part1_result, scenes_metadata,
        successful_encodings, output_dir, pipeline_start_time,
        part1_time, part2_time, part3_time, part4_time, total_pipeline_time,
        final_vmaf, comprehensive_report
    )
    
    return final_video_path


def test_video_compression(video_path: str) -> Optional[str]:
    """
    Test function for video compression using default parameters.
    
    Args:
        video_path: Path to input video file
        
    Returns:
        str: Path to compressed video file, or None if failed
    """
    print(f"\n🧪 === Testing Video Compression ===")
    print(f"   📁 Input: {video_path}")
    print(f"   🎯 Using default test parameters")
    
    # Default test parameters
    test_params = {
        'target_quality': 'Medium',
        'max_duration': 3600,
        'output_dir': './test_output'
    }
    
    try:
        # Validate input file
        input_path = Path(video_path)
        if not input_path.is_file():
            print(f"❌ Input file does not exist: {video_path}")
            return None
        # Create test output directory
        test_output_dir = Path(test_params['output_dir'])
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Perform compression
        result = video_compressor(
            input_file=str(input_path),
            target_quality=test_params['target_quality'],
            max_duration=test_params['max_duration'],
            output_dir=str(test_output_dir)
        )
        
        if result and Path(result).exists():
            print(f"\n✅ Test completed successfully!")
            print(f"   📁 Compressed video: {result}")
            return result
        else:
            print(f"\n❌ Test failed - no output file generated")
            return None
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        return None


# ============================================================================
# Helper Functions
# ============================================================================

def _load_configuration(current_dir: str) -> dict:
    """Load configuration from config.json or use defaults."""
    try:
        config_path = os.path.join(current_dir, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("✅ Configuration loaded successfully")
        return config
    except FileNotFoundError:
        print("⚠️ Config file not found, using default configuration")
        return _get_default_config()


def _get_default_config() -> dict:
    """Get default configuration when config.json is not available."""
    return {
        'directories': {
            'temp_dir': './videos/temp_scenes',
            'output_dir': './output'
        },
        'video_processing': {
            'SHORT_VIDEO_THRESHOLD': 20,
            'target_vmaf': 93.0,
            'codec': 'auto',  # Legacy parameter, overridden by target_codec
            'target_codec': 'av1_nvenc',  # Will be overridden by request
            'codec_mode': 'CRF',  # Will be overridden by request
            'target_bitrate': 10.0,  # Will be overridden by request
            'size_increase_protection': True,
            'conservative_cq_adjustment': 0,
            'max_output_size_ratio': 1.15,
            'max_encoding_retries': 2,
            'basic_cq_lookup_by_quality': {
                'High': {
                    'animation': 44,
                    'low-action': 40,
                    'medium-action': 38,
                    'high-action': 34,
                    'default': 40
                },
                'Medium': {
                    'animation': 46,
                    'low-action': 42,
                    'medium-action': 40,
                    'high-action': 36,
                    'default': 42
                },
                'Low': {
                    'animation': 52,
                    'low-action': 48,
                    'medium-action': 46,
                    'high-action': 42,
                    'default': 48
                }
            },
        },
        'scene_detection': {
            'enable_time_based_fallback': True,
            'time_based_scene_duration': 90
        },
        'vmaf_calculation': {
            'calculate_full_video_vmaf': True,
            'vmaf_use_sampling': True,
            'vmaf_num_clips': 3,
            'vmaf_clip_duration': 2
        },
        'output_settings': {
            'save_individual_scene_reports': True,
            'save_comprehensive_report': True
        },
        'model_paths': {
            'scene_classifier_model': 'services/compress/models/scene_classifier_model.pth'
        }
    }


def _display_pipeline_info(input_file: str, target_quality: str, max_duration: int, output_dir: str):
    """Display pipeline initialization information."""
    print(f"\n🎬 === AI Video Compression Pipeline ===")
    print(f"   📁 Input: {Path(input_file).name}")
    print(f"   🎯 Target Quality: {target_quality}")
    print(f"   ⏱️ Max Duration: {max_duration}s")
    print(f"   📁 Output Dir: {output_dir}")
    print(f"   🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def _execute_preprocessing(input_file: str, target_quality: str, max_duration: int, output_dir_path: Path) -> Optional[dict]:
    """Execute Part 1: Pre-processing."""
    print(f"\n🔧 === Part 1: Pre-processing ===")
    part1_start_time = time.time()
    
    part1_result = pre_processing(
        video_path=input_file,
        target_quality=target_quality,
        max_duration=max_duration,
        output_dir=output_dir_path
    )
    
    print("part1_result", part1_result)
    return part1_result


def _display_preprocessing_results(part1_result: dict, part1_time: float):
    """Display Part 1 results."""
    print(f"\n✅ Part 1 completed in {part1_time:.1f}s:")
    print(f"   📁 Video: {os.path.basename(part1_result['path'])}")
    print(f"   🎥 Codec: {part1_result['codec']} (original: {part1_result['original_codec']})")
    print(f"   ⏱️ Duration: {part1_result['duration']:.1f}s")
    print(f"   🔄 Reencoded: {part1_result['was_reencoded']}")
    print(f"   🎯 Target VMAF: {part1_result['target_vmaf']} ({part1_result['target_quality']})")
    
    if part1_result['was_reencoded']:
        print(f"   🔄 Lossless conversion: {part1_result['processing_info']['original_format']} → {part1_result['processing_info']['standardized_format']}")
        print(f"   ⏱️ Encoding time: {part1_result['encoding_time']:.1f}s")


def _display_scene_detection_results(scenes_metadata: list, part2_time: float):
    """Display Part 2 results."""
    print(f"\n✅ Part 2 completed in {part2_time:.1f}s: {len(scenes_metadata)} scenes detected")
    
    # Display scene information
    total_scene_size = 0
    for scene in scenes_metadata:
        scene_size = scene.get('file_size_mb', 0)
        total_scene_size += scene_size
        print(f"   Scene {scene['scene_number']}: {scene['start_time']:.1f}s - {scene['end_time']:.1f}s "
              f"(duration: {scene['duration']:.1f}s)")
        if scene_size > 0:
            print(f"      📁 File: {os.path.basename(scene['path'])} ({scene_size:.1f} MB)")
        else:
            print(f"      📁 File: {os.path.basename(scene['path'])}")
    
    if total_scene_size > 0:
        print(f"   📊 Total scene files: {total_scene_size:.1f} MB")


def _execute_ai_encoding(scenes_metadata: list, config: dict, target_quality: str,
                         encode_budget: float = 120.0) -> Optional[dict]:
    """Execute Part 3: AI Encoding."""
    print(f"\n🧠 === Part 3: AI Encoding (budget: {encode_budget:.0f}s) ===")
    part3_start_time = time.time()

    TOTAL_ENCODE_BUDGET = encode_budget

    vmaf_threshold = {'High': 93.0, 'Medium': 89.0, 'Low': 85.0}.get(target_quality, 89.0)
    codec = config.get('video_processing', {}).get('target_codec', 'av1_nvenc')
    codec_mode = config.get('video_processing', {}).get('codec_mode', 'CRF')
    target_bitrate = config.get('video_processing', {}).get('target_bitrate', 10.0)

    print(f"   🎯 Target: VMAF {vmaf_threshold} | Codec: {codec} | Mode: {codec_mode}")
    print(f"   🧠 Method: VMAF-targeted iterative binary search")

    try:
        resources = load_encoding_resources(config, logging_enabled=True)
        print(f"   ✅ AI resources loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load AI resources: {e}")
        return None

    # Process each scene individually
    encoded_scenes_data = []
    successful_encodings = 0
    failed_encodings = 0
    total_input_size = 0
    total_output_size = 0
    total_scenes = len(scenes_metadata)

    print(f"\n   📊 Processing {total_scenes} scenes with iterative encoder...")

    for i, scene_metadata in enumerate(scenes_metadata):
        # Calculate per-scene time budget from remaining time
        elapsed = time.time() - part3_start_time
        remaining_budget = TOTAL_ENCODE_BUDGET - elapsed
        remaining_scenes = total_scenes - i
        scene_time_budget = remaining_budget / remaining_scenes

        scene_result = _process_single_scene(
            scene_metadata, i, total_scenes, config, resources, target_quality,
            scene_time_budget=scene_time_budget,
            vmaf_threshold=vmaf_threshold,
            codec=codec,
            codec_mode=codec_mode,
            target_bitrate=target_bitrate,
        )

        if scene_result['success']:
            successful_encodings += 1
            total_input_size += scene_result['input_size_mb']
            total_output_size += scene_result['output_size_mb']
        else:
            failed_encodings += 1

        encoded_scenes_data.append(scene_result['scene_data'])
    
    part3_time = time.time() - part3_start_time
    
    # Display Part 3 summary
    _display_ai_encoding_summary(
        successful_encodings, failed_encodings, len(scenes_metadata),
        part3_time, target_quality, total_input_size, total_output_size
    )
    
    if successful_encodings == 0:
        print("❌ Part 3 failed completely. No scenes were encoded. Pipeline terminated.")
        return None
    
    return {
        'encoded_scenes_data': encoded_scenes_data,
        'successful_encodings': successful_encodings,
        'failed_encodings': failed_encodings,
        'processing_time': part3_time,
        'total_input_size': total_input_size,
        'total_output_size': total_output_size
    }


def _process_single_scene(scene_metadata: dict, scene_index: int, total_scenes: int,
                         config: dict, resources: dict, target_quality: str,
                         scene_time_budget: float = 120.0,
                         vmaf_threshold: float = 89.0,
                         codec: str = 'av1_nvenc',
                         codec_mode: str = 'CRF',
                         target_bitrate: float = 10.0) -> dict:
    """Process a single scene using VMAF-targeted iterative encoding with content analysis."""
    scene_number = scene_metadata['scene_number']
    scene_path = scene_metadata['path']
    scene_duration = scene_metadata['duration']

    print(f"\n   🎬 Scene {scene_number}/{total_scenes}: {os.path.basename(scene_path)}")
    print(f"      ⏱️ Duration: {scene_duration:.1f}s | Budget: {scene_time_budget:.0f}s")
    print(f"      🎯 VMAF target: {vmaf_threshold} | Codec: {codec} | Mode: {codec_mode}")

    scene_start_time = time.time()

    # Content analysis: classify scene and extract contrast for content-aware encoding
    scene_type = None
    contrast_value = None
    if resources and resources.get('scene_classifier_model'):
        try:
            from utils.processing_utils import classify_scene_from_path
            temp_dir = config.get('directories', {}).get('temp_dir', './videos/temp_scenes')
            classification_result = classify_scene_from_path(
                scene_path=scene_path,
                temp_dir=temp_dir,
                scene_classifier_model=resources['scene_classifier_model'],
                available_metrics=resources.get('available_metrics', []),
                device=resources.get('device', 'cpu'),
                metrics_scaler=resources.get('feature_scaler_step'),
                class_mapping=resources.get('class_mapping'),
                logging_enabled=False,
                num_frames=3,
            )
            if isinstance(classification_result, tuple) and len(classification_result) >= 2:
                scene_type = classification_result[0]
                video_features = classification_result[2] if len(classification_result) >= 3 else {}
                contrast_value = video_features.get('contrast', video_features.get('metrics_avg_contrast'))
            analysis_time = time.time() - scene_start_time
            print(f"      🎭 Scene: '{scene_type}' | Contrast: {contrast_value:.2f if contrast_value else 'N/A'} "
                  f"({analysis_time:.1f}s)")
        except Exception as e:
            print(f"      ⚠️ Scene classification failed ({e}), using defaults")

    # Determine output path
    output_dir = config.get('directories', {}).get('temp_dir', './videos/temp_scenes')
    base_name = os.path.splitext(os.path.basename(scene_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_encoded.mp4")

    # Reduce time budget by analysis time
    elapsed_analysis = time.time() - scene_start_time
    adjusted_budget = scene_time_budget - elapsed_analysis

    try:
        result = iterative_encode(
            input_path=scene_path,
            output_path=output_path,
            codec=codec,
            vmaf_threshold=vmaf_threshold,
            codec_mode=codec_mode,
            target_bitrate=target_bitrate,
            time_budget=adjusted_budget,
            scene_type=scene_type,
            contrast_value=contrast_value,
        )

        scene_processing_time = time.time() - scene_start_time

        if result['success'] and os.path.exists(output_path):
            input_size_mb = os.path.getsize(scene_path) / (1024 * 1024)
            output_size_mb = os.path.getsize(output_path) / (1024 * 1024)

            print(f"      ✅ Scene {scene_number} encoded successfully")
            print(f"         📊 Size: {input_size_mb:.1f} MB → {output_size_mb:.1f} MB "
                  f"({result['compression_ratio']:.1f}x)")
            if result.get('vmaf') is not None:
                print(f"         🎯 VMAF: {result['vmaf']:.2f} (threshold: {vmaf_threshold})")
            print(f"         🎚️ Rate: {result['rate']} | Iterations: {result['iterations']}"
                  f"{' (fallback)' if result.get('fallback') else ''}")
            print(f"         ⏱️ Processing: {scene_processing_time:.1f}s")

            scene_data = {
                'scene_number': scene_number,
                'encoding_success': True,
                'encoded_file_size_mb': output_size_mb,
                'input_size_mb': input_size_mb,
                'compression_ratio': result['compression_ratio'],
                'vmaf_score': result.get('vmaf'),
                'final_rate': result['rate'],
                'iterations': result['iterations'],
                'fallback': result.get('fallback', False),
                'processing_time_seconds': scene_processing_time,
                'encoded_path': output_path,
                'original_video_metadata': scene_metadata['original_video_metadata'],
                'target_quality_level': target_quality,
            }

            scene_metadata['encoded_path'] = output_path
            scene_metadata['encoding_data'] = scene_data

            return {
                'success': True,
                'scene_data': scene_data,
                'input_size_mb': input_size_mb,
                'output_size_mb': output_size_mb,
            }
        else:
            # Iterative encoder produced no output — report failure
            # Do NOT fall back to ai_encoding (takes 60-90s, causes overtime)
            scene_processing_time = time.time() - scene_start_time
            print(f"      ❌ Scene {scene_number} iterative encoder failed, no output")
            print(f"         ⏱️ Processing: {scene_processing_time:.1f}s")

            error_scene_data = {
                'scene_number': scene_number,
                'encoding_success': False,
                'error_reason': 'Iterative encoder failed to produce output',
                'processing_time_seconds': scene_processing_time,
                'encoded_path': None,
                'original_video_metadata': scene_metadata['original_video_metadata'],
            }
            scene_metadata['encoded_path'] = None
            scene_metadata['encoding_data'] = error_scene_data
            return {
                'success': False,
                'scene_data': error_scene_data,
                'input_size_mb': 0,
                'output_size_mb': 0,
            }

    except Exception as e:
        scene_processing_time = time.time() - scene_start_time
        print(f"      ❌ Scene {scene_number} failed: {e}")
        print(f"         ⏱️ Processing: {scene_processing_time:.1f}s")

        error_scene_data = {
            'scene_number': scene_number,
            'encoding_success': False,
            'error_reason': f'Exception: {str(e)}',
            'processing_time_seconds': scene_processing_time,
            'encoded_path': None,
            'original_video_metadata': scene_metadata['original_video_metadata']
        }

        scene_metadata['encoded_path'] = None
        scene_metadata['encoding_data'] = error_scene_data

        return {
            'success': False,
            'scene_data': error_scene_data,
            'input_size_mb': 0,
            'output_size_mb': 0,
        }


def _display_ai_encoding_summary(successful_encodings: int, failed_encodings: int, total_scenes: int,
                                part3_time: float, target_quality: str, total_input_size: float, total_output_size: float):
    """Display Part 3 summary."""
    print(f"\n   📊 Part 3 Processing Summary:")
    print(f"      ✅ Successful encodings: {successful_encodings}")
    print(f"      ❌ Failed encodings: {failed_encodings}")
    print(f"      📈 Success rate: {successful_encodings/total_scenes*100:.1f}%")
    print(f"      ⏱️ Total processing time: {part3_time:.1f}s")
    print(f"      🎯 Quality Level: {target_quality}")
    print(f"      🧠 AI Method: Scene classification + quality-based CQ lookup")
    
    if total_input_size > 0 and total_output_size > 0:
        overall_compression = (1 - total_output_size / total_input_size) * 100
        print(f"      🗜️ Overall compression: {overall_compression:+.1f}%")
        print(f"      📊 Total size: {total_input_size:.1f} MB → {total_output_size:.1f} MB")
    
    print(f"✅ Part 3 completed with {successful_encodings} successful encodings")


def _execute_validation_and_merging(part1_result: dict, encoded_scenes_data: list, config: dict) -> Optional[dict]:
    """Execute Part 4: Validation and Merging."""
    part4_start_time = time.time()
    
    try:
        final_video_path, final_vmaf, comprehensive_report = validation_and_merging(
            original_video_path=part1_result['path'],
            encoded_scenes_data=encoded_scenes_data,
            config=config,
            logging_enabled=True
        )
        
        part4_time = time.time() - part4_start_time
        
        if final_video_path and os.path.exists(final_video_path):
            return {
                'final_video_path': final_video_path,
                'final_vmaf': final_vmaf,
                'comprehensive_report': comprehensive_report,
                'processing_time': part4_time
            }
        else:
            print("❌ Part 4 failed. Could not create final video.")
            return None
            
    except Exception as e:
        print(f"❌ Part 4 failed with exception: {e}")
        return None


def _display_validation_results(part4_result: dict):
    """Display Part 4 results."""
    final_video_path = part4_result['final_video_path']
    final_vmaf = part4_result['final_vmaf']
    comprehensive_report = part4_result['comprehensive_report']
    part4_time = part4_result['processing_time']
    
    print(f"✅ Part 4 completed successfully in {part4_time:.1f}s!")
    print(f"   📁 Final video: {os.path.basename(final_video_path)}")
    
    if final_vmaf:
        print(f"   🎯 Final VMAF: {final_vmaf:.2f}")
    
    if comprehensive_report:
        compression_info = comprehensive_report.get('compression_metrics', {})
        final_compression = compression_info.get('overall_compression_ratio_percent', 0)
        final_size = compression_info.get('final_file_size_mb', 0)
        
        print(f"   🗜️ Overall compression: {final_compression:+.1f}%")
        print(f"   📊 Final file size: {final_size:.1f} MB")


def _display_pipeline_summary(input_file: str, final_video_path: str, part1_result: dict,
                            scenes_metadata: list, successful_encodings: int, output_dir: str,
                            pipeline_start_time: float, part1_time: float, part2_time: float,
                            part3_time: float, part4_time: float, total_pipeline_time: float,
                            final_vmaf: Optional[float], comprehensive_report: Optional[dict]):
    """Display complete pipeline summary."""
    print(f"\n🎉 === Pipeline Completed Successfully ===")
    print(f"   📁 Input video: {os.path.basename(input_file)}")
    print(f"   📁 Final video: {os.path.basename(final_video_path)}")
    print(f"   🎯 Target quality: {part1_result['target_quality']} (VMAF: {part1_result['target_vmaf']})")
    print(f"   📊 Scenes processed: {len(scenes_metadata)} total, {successful_encodings} successful")
    print(f"   📁 Output directory: {output_dir}")
    print(f"   🕐 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Performance breakdown
    print(f"\n   ⏱️ Performance Breakdown:")
    print(f"      Part 1 (Pre-processing): {part1_time:.1f}s")
    print(f"      Part 2 (Scene Detection): {part2_time:.1f}s")
    print(f"      Part 3 (AI Encoding): {part3_time:.1f}s")
    print(f"      Part 4 (Validation & Merging): {part4_time:.1f}s")  
    print(f"      Total Pipeline Time: {total_pipeline_time:.1f}s")
    
    # Final file size comparison
    input_file_path = Path(input_file)
    final_video_path_obj = Path(final_video_path)
    if input_file_path.exists() and final_video_path_obj.exists():
        input_size = input_file_path.stat().st_size / (1024 * 1024)
        output_size = final_video_path_obj.stat().st_size / (1024 * 1024)
        final_compression = (1 - output_size / input_size) * 100
        
        print(f"\n   📊 Final Size Comparison:")
        print(f"      Input: {input_size:.1f} MB")
        print(f"      Output: {output_size:.1f} MB")
        print(f"      Compression: {final_compression:+.1f}%")
        
        if final_compression > 0:
            print(f"      💾 Space saved: {input_size - output_size:.1f} MB")
    
    # Quality achievement summary
    if final_vmaf and comprehensive_report:
        quality_info = comprehensive_report.get('quality_metrics', {})
        scenes_meeting_target = quality_info.get('scenes_meeting_target', 0)
        avg_scene_vmaf = quality_info.get('average_scene_vmaf', 0)
        
        print(f"\n   🎯 Quality Achievement:")
        print(f"      Final VMAF: {final_vmaf:.2f}")
        print(f"      Average Scene VMAF: {avg_scene_vmaf:.2f}")
        print(f"      Scenes meeting target: {scenes_meeting_target}/{len(scenes_metadata)}")
        
        if 'prediction_accuracy_stats' in comprehensive_report.get('scene_analysis', {}):
            pred_stats = comprehensive_report['scene_analysis']['prediction_accuracy_stats']
            avg_error = pred_stats.get('average_prediction_error')
            if avg_error:
                print(f"      AI prediction accuracy: ±{avg_error:.1f} VMAF points")
    
    # Report file locations
    if comprehensive_report:
        print(f"\n   📄 Reports Generated:")
        print(f"      📁 Output directory: {output_dir}")
        print(f"      📊 Comprehensive report: comprehensive_processing_report_*.json")
        print(f"      📄 Individual scene reports: scene_reports/scene_*_report.json")
    
    print(f"\n   🎉 Pipeline completed successfully!")
    print(f"   🚀 Ready for playback: {final_video_path}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # 3 workers: handles concurrent validator requests without queueing.
    # AV1 uses GPU (NVENC supports 8 sessions), HEVC uses CPU (32 cores available).
    WORKERS = 3

    logger.info(f"Starting video compressor server ({WORKERS} workers)")
    logger.info(f"Video compressor server running on http://{CONFIG.video_compressor.host}:{CONFIG.video_compressor.port}")

    uvicorn.run(
        "server:app",
        host=CONFIG.video_compressor.host,
        port=CONFIG.video_compressor.port,
        workers=WORKERS,
    )

    # result = test_video_compression('test1.mp4')
    # print(result)

    #python services/compress/server.py