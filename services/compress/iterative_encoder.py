"""
VMAF-targeted iterative encoder for Vidaio SN85 compression.

Strategy:
1. Calibrate with fast 2-second clip encodes to find approximate threshold CQ
2. Binary search with full encodes + VMAF to refine to the boundary
3. Always pushes for maximum compression above VMAF threshold

The `rate` parameter passed to encode_video() is always in av1_nvenc CQ units.
For other codecs (libx265, etc.), apply_rate_mapping() translates automatically.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from utils.encode_video import encode_video


REPO_DIR = Path(__file__).resolve().parent.parent.parent.parent
FFMPEG_VMAF = str(REPO_DIR / "tools" / "ffmpeg-vmaf")
VMAF_NEG_MODEL = str(
    REPO_DIR / "upstream" / "venv" / "lib" / "python3.12"
    / "site-packages" / "ffmpeg_quality_metrics" / "vmaf_models"
    / "vmaf_v0.6.1neg.json"
)

QUALITY_MAP = {'High': 93.0, 'Medium': 89.0, 'Low': 85.0}

# Hard timeout is 135s. Budget = 135 - 5 = 130s.
SAFETY_MARGIN_SECONDS = 5.0

# CQ limits for av1_nvenc (0-51 effective range)
AV1_CQ_MAX = 51
AV1_CQ_MIN = 15

# Validator measures VMAF with n_subsample=1 (every frame) + harmonic mean.
# Sparse sampling misses outlier frames, so our measurement reads higher.
# Use tighter sampling for high thresholds where crossing matters most.
VMAF_SUBSAMPLE = {93: 10, 89: 20, 85: 30}

# Safety margin was counterproductive: targeting VMAF 95 for T93 caused the
# binary search to oscillate (first iter at CQ 32 → VMAF 96 with 1.7x
# compression, then overshooting to CQ 41 → VMAF 91). With n_subsample=10
# the measurement is accurate enough to target the real threshold directly.
VMAF_SAFETY_MARGIN = {93: 0.0, 89: 0.0, 85: 0.0}


def _find_ffmpeg() -> str:
    if os.path.isfile(FFMPEG_VMAF) and os.access(FFMPEG_VMAF, os.X_OK):
        return FFMPEG_VMAF
    return "ffmpeg"


def _harmonic_mean(scores: list[float]) -> float:
    if not scores:
        return 0.0
    return len(scores) / sum(1.0 / max(s, 0.01) for s in scores)


def measure_vmaf_fast(reference_path: str, distorted_path: str, n_subsample: int = 15) -> float:
    """Measure VMAF with subsampling for speed. Returns -1 on error."""
    ffmpeg = _find_ffmpeg()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        vmaf_log = tmp.name

    try:
        cmd = [
            ffmpeg,
            "-i", distorted_path,
            "-i", reference_path,
            "-lavfi",
            (
                f"libvmaf="
                f"log_fmt=json:"
                f"log_path={vmaf_log}:"
                f"model='path={VMAF_NEG_MODEL}':"
                f"n_subsample={n_subsample}"
            ),
            "-f", "null",
            "-",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            print(f"[iterative] VMAF ffmpeg failed (exit {result.returncode})")
            return -1.0

        if not os.path.exists(vmaf_log) or os.path.getsize(vmaf_log) == 0:
            return -1.0

        with open(vmaf_log) as f:
            data = json.load(f)

        scores = [
            fr["metrics"]["vmaf"]
            for fr in data.get("frames", [])
            if fr.get("metrics", {}).get("vmaf", 0) > 0
        ]

        if not scores:
            return -1.0

        return _harmonic_mean(scores)

    except subprocess.TimeoutExpired:
        print("[iterative] VMAF measurement timed out")
        return -1.0
    except Exception as e:
        print(f"[iterative] VMAF error: {e}")
        return -1.0
    finally:
        if os.path.exists(vmaf_log):
            os.unlink(vmaf_log)


def _get_video_info(input_path: str) -> tuple[float, float]:
    """Get video duration (seconds) and bitrate (Mbps)."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-show_entries",
            "format=duration,bit_rate", "-of", "json", input_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            fmt = data.get("format", {})
            duration = float(fmt.get("duration", 10))
            bitrate_mbps = float(fmt.get("bit_rate", 50_000_000)) / 1_000_000
            return duration, bitrate_mbps
    except Exception:
        pass
    return 10.0, 50.0


def _extract_clip(input_path: str, clip_path: str, duration: float, clip_length: float = 2.0):
    """Extract a short clip from the middle of the video."""
    start = max(0, duration / 2 - clip_length / 2)
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-i", input_path,
         "-t", str(clip_length), "-c", "copy", clip_path],
        capture_output=True, timeout=10
    )


def _encode_clip(clip_path: str, output_path: str, codec: str, cq: int):
    """Fast-encode a clip at a specific CQ. Returns file size in bytes, or 0 on failure."""
    if codec == "av1_nvenc":
        cmd = ["ffmpeg", "-y", "-i", clip_path, "-c:v", "av1_nvenc", "-preset", "p5",
               "-cq", str(cq), "-rc", "vbr", "-pix_fmt", "yuv420p", "-an", output_path]
    elif codec == "libx265":
        cmd = ["ffmpeg", "-y", "-i", clip_path, "-c:v", "libx265", "-preset", "fast",
               "-crf", str(cq), "-pix_fmt", "yuv420p", "-an", output_path]
    else:
        cmd = ["ffmpeg", "-y", "-i", clip_path, "-c:v", codec, "-preset", "fast",
               "-crf", str(cq), "-pix_fmt", "yuv420p", "-an", output_path]

    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode == 0 and os.path.exists(output_path):
        return os.path.getsize(output_path)
    return 0


def calibrate_with_clips(input_path: str, codec: str, vmaf_threshold: float,
                         duration: float, input_size: int) -> tuple:
    """
    Fast clip-based calibration to find optimal starting CQ.

    Encodes 2-second clips at multiple CQ values (~1s each), then
    interpolates to find a good starting point for the binary search.

    Returns (rate_low, rate_high, start_rate) or None.
    """
    clip_path = None
    clip_outputs = []

    try:
        clip_path = tempfile.mktemp(suffix="_clip.mp4")
        _extract_clip(input_path, clip_path, duration)

        if not os.path.exists(clip_path) or os.path.getsize(clip_path) == 0:
            return None

        clip_size = os.path.getsize(clip_path)

        test_cqs = [35, 40, 45, 50] if codec == "av1_nvenc" else [30, 37, 44, 50]
        results = []

        for cq in test_cqs:
            out = tempfile.mktemp(suffix=f"_cq{cq}.mp4")
            clip_outputs.append(out)
            enc_size = _encode_clip(clip_path, out, codec, cq)
            if enc_size > 0:
                ratio = clip_size / enc_size
                results.append((cq, ratio))

        if len(results) < 2:
            return None

        results.sort(key=lambda x: x[0])

        # Adaptive target: aim for 60% of max achievable compression.
        # This ensures we start above threshold (conservative) while being
        # aggressive enough to push the boundary quickly.
        max_ratio = results[-1][1]
        target_ratio = max_ratio * 0.6

        # Interpolate to find CQ that gives target_ratio
        start_cq = results[len(results) // 2][0]  # fallback: midpoint
        for i in range(len(results) - 1):
            cq_lo, ratio_lo = results[i]
            cq_hi, ratio_hi = results[i + 1]
            if ratio_lo <= target_ratio <= ratio_hi:
                t = (target_ratio - ratio_lo) / (ratio_hi - ratio_lo)
                start_cq = int(cq_lo + t * (cq_hi - cq_lo))
                break

        refined_low = max(results[0][0] - 3, AV1_CQ_MIN)
        refined_high = min(results[-1][0] + 2, AV1_CQ_MAX)

        return refined_low, refined_high, start_cq

    except Exception as e:
        print(f"[iterative] Calibration failed: {e}")
        return None
    finally:
        if clip_path and os.path.exists(clip_path):
            os.unlink(clip_path)
        for out in clip_outputs:
            if os.path.exists(out):
                os.unlink(out)


def _validate_vbr_bitrate(output_path: str, target_bitrate: float) -> bool:
    """Check that output average bitrate doesn't exceed target * 1.10."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", "format=bit_rate",
            "-of", "csv=p=0", output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            actual_mbps = float(result.stdout.strip()) / 1_000_000
            return actual_mbps <= target_bitrate * 1.10
    except Exception:
        pass
    return True


def iterative_encode(
    input_path: str,
    output_path: str,
    codec: str,
    vmaf_threshold: float,
    codec_mode: str = "CRF",
    target_bitrate: float = 10.0,
    time_budget: float = 130.0,
    scene_type: str = None,
    contrast_value: float = None,
) -> dict:
    """
    Iteratively encode to maximize compression while meeting VMAF threshold.

    Uses content-aware parameters when scene_type and contrast_value are
    provided (from upstream scene classification + video analysis).

    Args:
        input_path: Source video path.
        output_path: Desired output path.
        codec: FFmpeg encoder name (e.g., 'av1_nvenc', 'libx265').
        vmaf_threshold: Minimum acceptable VMAF score.
        codec_mode: 'CRF', 'VBR', or 'CBR'.
        target_bitrate: Bitrate in Mbps (for VBR/CBR).
        time_budget: Seconds available.
        scene_type: Content classification (e.g., 'Faces / People', 'Gaming Content').
        contrast_value: Perceptual contrast 0.0-1.0 for AQ tuning.

    Returns:
        dict with keys: success, vmaf, rate, compression_ratio, compression_rate,
                        iterations, time, fallback
    """
    input_size = os.path.getsize(input_path)
    duration, input_mbps = _get_video_info(input_path)

    start_time = time.time()

    # Content-aware CQ lookup: use scene classification to pick a starting
    # CQ that matches the content type, instead of a flat empirical value.
    # Mapping mirrors upstream encoder.py map_scene_type_to_lookup_key().
    _SCENE_TO_ACTION = {
        'screen content / text': 'low-action',
        'animation / cartoon / rendered graphics': 'animation',
        'faces / people': 'low-action',
        'gaming content': 'high-action',
        'other': 'medium-action',
        'unclear': 'default',
        'default': 'default',
    }
    _CQ_LOOKUP = {
        'high': {'animation': 42, 'low-action': 38, 'medium-action': 36, 'high-action': 32, 'default': 38},
        'medium': {'animation': 46, 'low-action': 42, 'medium-action': 40, 'high-action': 36, 'default': 42},
        'low': {'animation': 50, 'low-action': 46, 'medium-action': 44, 'high-action': 40, 'default': 46},
    }

    threshold_int = int(vmaf_threshold)
    quality_tier = 'high' if threshold_int >= 93 else 'medium' if threshold_int >= 88 else 'low'

    if scene_type:
        action_key = _SCENE_TO_ACTION.get(scene_type.lower(), 'default')
        rate = _CQ_LOOKUP[quality_tier].get(action_key, _CQ_LOOKUP[quality_tier]['default'])
        print(f"[iterative] Content-aware start: scene='{scene_type}' → "
              f"action='{action_key}' tier='{quality_tier}' → rate={rate}")
    else:
        EMPIRICAL_STARTS = {85: 49, 89: 46, 93: 40}
        rate = EMPIRICAL_STARTS.get(threshold_int, 45)
        print(f"[iterative] Fallback start (no scene type): rate={rate}")

    rate_low = max(rate - 15, AV1_CQ_MIN)
    rate_high = AV1_CQ_MAX

    n_subsample = VMAF_SUBSAMPLE.get(threshold_int, 30)
    safety_margin = VMAF_SAFETY_MARGIN.get(threshold_int, 0.0)
    effective_threshold = vmaf_threshold + safety_margin

    print(f"[iterative] Empirical start: rate={rate} range=[{rate_low},{rate_high}]")
    print(f"[iterative] VMAF: threshold={vmaf_threshold} effective={effective_threshold} "
          f"n_subsample={n_subsample} margin={safety_margin}")

    # VBR adjustment: push slightly more aggressive since bitrate cap protects us.
    # Skip for T93 — the threshold is too tight and the +3 pushes into VMAF 86-88.
    if codec_mode.upper() == "VBR" and threshold_int < 93:
        rate = min(rate + 3, rate_high - 3)

    # Phase 2: Binary search with full encodes + VMAF
    best_result = None
    iteration = 0
    temp_files = []

    print(f"[iterative] Binary search: codec={codec} threshold={vmaf_threshold} "
          f"range=[{rate_low}, {rate_high}] start={rate} budget={time_budget:.0f}s")

    try:
        while True:
            elapsed = time.time() - start_time
            remaining = time_budget - elapsed

            if remaining < SAFETY_MARGIN_SECONDS:
                print(f"[iterative] Time budget exhausted after {iteration} iterations")
                break

            # Allow starting if we have ~85% of avg iteration time
            if iteration > 0:
                avg_iter_time = elapsed / iteration
                if remaining < avg_iter_time * 0.85:
                    print(f"[iterative] Not enough time for another iteration "
                          f"(need ~{avg_iter_time:.0f}s, have {remaining:.0f}s)")
                    break

            iteration += 1
            temp_output = f"{output_path}.iter{iteration}.mp4"
            temp_files.append(temp_output)

            print(f"[iterative] Iteration {iteration}: rate={rate} "
                  f"(range [{rate_low}, {rate_high}]) elapsed={elapsed:.1f}s")

            # Encode with fast preset for search speed
            # HEVC uses 'veryfast' not 'superfast' — superfast produces significantly
            # lower VMAF per byte, causing T89/T93 to miss threshold
            search_preset = 'p5' if codec.endswith('_nvenc') else 'veryfast'
            encode_result = encode_video(
                input_path, temp_output, codec,
                rate=rate,
                preset=search_preset,
                scene_type=scene_type,
                contrast_value=contrast_value,
                codec_mode=codec_mode,
                target_bitrate=target_bitrate,
                logging_enabled=False
            )

            # encode_video returns (None, None) on hard failure.
            if encode_result is None or encode_result == (None, None):
                print(f"[iterative] Encode hard-failed at rate={rate}, moving conservative")
                rate_high = rate
                rate = (rate + rate_low) // 2
                if os.path.exists(temp_output):
                    os.unlink(temp_output)
                    temp_files.remove(temp_output)
                if abs(rate_high - rate_low) <= 2:
                    break
                continue

            if not os.path.exists(temp_output) or os.path.getsize(temp_output) == 0:
                print(f"[iterative] Output missing/empty at rate={rate}")
                rate_high = rate
                rate = (rate + rate_low) // 2
                if abs(rate_high - rate_low) <= 2:
                    break
                continue

            # VBR/CBR bitrate compliance
            if codec_mode.upper() in ("VBR", "CBR"):
                if not _validate_vbr_bitrate(temp_output, target_bitrate):
                    print(f"[iterative] Bitrate exceeded at rate={rate}, pushing more aggressive")
                    rate_low = rate
                    rate = (rate + rate_high) // 2
                    os.unlink(temp_output)
                    temp_files.remove(temp_output)
                    if abs(rate_high - rate_low) <= 2:
                        break
                    continue

            vmaf = measure_vmaf_fast(input_path, temp_output, n_subsample=n_subsample)
            if vmaf < 0:
                print(f"[iterative] VMAF measurement failed at rate={rate}")
                rate_high = rate
                rate = (rate + rate_low) // 2
                os.unlink(temp_output)
                temp_files.remove(temp_output)
                if abs(rate_high - rate_low) <= 2:
                    break
                continue

            output_size = os.path.getsize(temp_output)
            compression_ratio = input_size / output_size if output_size > 0 else 1.0
            compression_rate = output_size / input_size if input_size > 0 else 1.0

            print(f"[iterative]   → VMAF={vmaf:.2f} ratio={compression_ratio:.1f}x "
                  f"size={output_size / 1024 / 1024:.1f}MB")

            # Result classification uses the REAL threshold (what the validator scores
            # against), not the effective_threshold (which only guides binary search
            # direction). This prevents discarding results that ARE above the real
            # threshold just because they're below our safety margin.
            is_above = vmaf >= vmaf_threshold
            prev_above = best_result.get('above_threshold', False) if best_result else False

            if best_result is None:
                is_better = True
            elif is_above and not prev_above:
                is_better = True
            elif is_above and prev_above:
                is_better = compression_ratio > best_result['compression_ratio']
            elif not is_above and not prev_above:
                # Below threshold: prefer higher VMAF (soft-zone scoring is quadratic
                # in VMAF, so quality matters far more than compression here)
                is_better = vmaf > best_result['vmaf']
            else:
                is_better = False

            if is_better:
                if best_result and os.path.exists(best_result['temp_path']):
                    os.unlink(best_result['temp_path'])
                    if best_result['temp_path'] in temp_files:
                        temp_files.remove(best_result['temp_path'])
                best_result = {
                    'temp_path': temp_output,
                    'vmaf': vmaf,
                    'rate': rate,
                    'compression_ratio': compression_ratio,
                    'compression_rate': compression_rate,
                    'above_threshold': is_above,
                }
                if temp_output in temp_files:
                    temp_files.remove(temp_output)
            else:
                os.unlink(temp_output)
                temp_files.remove(temp_output)

            # Binary search — always push for maximum compression
            if vmaf >= effective_threshold:
                rate_low = rate
                rate = (rate + rate_high) // 2
            else:
                rate_high = rate
                rate = (rate + rate_low) // 2

            # Convergence
            if abs(rate_high - rate_low) <= 2:
                print(f"[iterative] Search converged (range={rate_high - rate_low})")
                break

        # Two-pass: re-encode at higher quality preset for better quality-per-byte.
        # p7 (AV1) or fast (HEVC) produces the same or higher VMAF at the same CQ,
        # sometimes with smaller file size due to better encoding efficiency.
        if best_result and best_result['above_threshold']:
            elapsed = time.time() - start_time
            remaining = time_budget - elapsed
            avg_iter = elapsed / max(iteration, 1)
            is_nvenc = codec.endswith('_nvenc')
            final_encode_est = avg_iter * (1.5 if is_nvenc else 0.7)

            if remaining > final_encode_est + SAFETY_MARGIN_SECONDS:
                final_preset = 'p7' if is_nvenc else 'fast'
                final_output = f"{output_path}.final.mp4"
                final_rate = best_result['rate']
                print(f"[iterative] Two-pass: re-encoding at {final_preset} rate={final_rate} "
                      f"(est {final_encode_est:.0f}s, have {remaining:.0f}s)")

                encode_result = encode_video(
                    input_path, final_output, codec,
                    rate=final_rate,
                    preset=final_preset,
                    scene_type=scene_type,
                    contrast_value=contrast_value,
                    codec_mode=codec_mode,
                    target_bitrate=target_bitrate,
                    logging_enabled=False
                )

                if (encode_result is not None and encode_result != (None, None)
                        and os.path.exists(final_output) and os.path.getsize(final_output) > 0):
                    final_size = os.path.getsize(final_output)
                    final_ratio = input_size / final_size
                    final_rate_val = final_size / input_size
                    print(f"[iterative] Two-pass encode: ratio={final_ratio:.1f}x "
                          f"size={final_size / 1024 / 1024:.1f}MB")

                    # Verify VMAF if time permits, otherwise trust that p7 >= p5 quality
                    elapsed2 = time.time() - start_time
                    remaining2 = time_budget - elapsed2
                    if remaining2 > 25:
                        vmaf_verify = measure_vmaf_fast(input_path, final_output, n_subsample=n_subsample)
                        print(f"[iterative] Two-pass verify: VMAF={vmaf_verify:.2f}")
                        if vmaf_verify >= vmaf_threshold:
                            os.unlink(best_result['temp_path'])
                            if best_result['temp_path'] in temp_files:
                                temp_files.remove(best_result['temp_path'])
                            best_result = {
                                'temp_path': final_output,
                                'vmaf': vmaf_verify,
                                'rate': final_rate,
                                'compression_ratio': final_ratio,
                                'compression_rate': final_rate_val,
                                'above_threshold': True,
                            }
                        else:
                            os.unlink(final_output)
                            print(f"[iterative] Two-pass below threshold, keeping search result")
                    else:
                        os.unlink(best_result['temp_path'])
                        if best_result['temp_path'] in temp_files:
                            temp_files.remove(best_result['temp_path'])
                        best_result = {
                            'temp_path': final_output,
                            'vmaf': best_result['vmaf'],
                            'rate': final_rate,
                            'compression_ratio': final_ratio,
                            'compression_rate': final_rate_val,
                            'above_threshold': True,
                        }
                        print(f"[iterative] Two-pass accepted (no time for verify)")
                else:
                    if os.path.exists(final_output):
                        os.unlink(final_output)
                    print(f"[iterative] Two-pass encode failed, keeping search result")
            else:
                print(f"[iterative] No time for two-pass ({remaining:.0f}s < {final_encode_est + SAFETY_MARGIN_SECONDS:.0f}s)")

        # Use best result
        if best_result:
            shutil.move(best_result['temp_path'], output_path)
            total_time = time.time() - start_time
            print(f"[iterative] Success: VMAF={best_result['vmaf']:.2f} "
                  f"ratio={best_result['compression_ratio']:.1f}x "
                  f"rate={best_result['rate']} iterations={iteration} "
                  f"time={total_time:.1f}s")
            return {
                'success': True,
                'vmaf': best_result['vmaf'],
                'rate': best_result['rate'],
                'compression_ratio': best_result['compression_ratio'],
                'compression_rate': best_result['compression_rate'],
                'iterations': iteration,
                'time': total_time,
                'fallback': False,
            }
        else:
            # Fallback: encode at most conservative rate if time allows
            elapsed = time.time() - start_time
            remaining = time_budget - elapsed
            est_encode_time = (elapsed / max(iteration, 1)) * 0.8
            if remaining > est_encode_time + 3:
                print(f"[iterative] No valid result, fallback at rate={rate_low}")
                encode_video(
                    input_path, output_path, codec,
                    rate=rate_low,
                    scene_type=scene_type,
                    contrast_value=contrast_value,
                    codec_mode=codec_mode,
                    target_bitrate=target_bitrate,
                    logging_enabled=False
                )
            else:
                print(f"[iterative] No valid result and no time for fallback")

            total_time = time.time() - start_time
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                output_size = os.path.getsize(output_path)
                return {
                    'success': True,
                    'vmaf': None,
                    'rate': rate_low,
                    'compression_ratio': input_size / output_size,
                    'compression_rate': output_size / input_size,
                    'iterations': iteration,
                    'time': total_time,
                    'fallback': True,
                }
            return {
                'success': False,
                'vmaf': None,
                'rate': rate_low,
                'compression_ratio': 1.0,
                'compression_rate': 1.0,
                'iterations': iteration,
                'time': total_time,
                'fallback': True,
            }

    finally:
        for temp in temp_files:
            if os.path.exists(temp):
                try:
                    os.unlink(temp)
                except OSError:
                    pass
