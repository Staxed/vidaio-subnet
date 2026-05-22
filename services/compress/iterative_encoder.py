"""
Data-driven single-shot encoder for Vidaio SN85 compression.

Strategy:
1. Pick a conservative CQ from an empirically-derived lookup table
   (built from 15 days of production convergence data)
2. Encode once, measure VMAF once
3. If above threshold → ship it (~35s total)
4. If below → rescue re-encode at lower CQ, skip second VMAF (~45s total)

The `rate` parameter passed to encode_video() is always in av1_nvenc CQ units.
For other codecs (libx265, etc.), apply_rate_mapping() translates automatically.
"""

import json
import math
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
    elif codec == "libsvtav1":
        cmd = ["ffmpeg", "-y", "-i", clip_path, "-c:v", "libsvtav1", "-preset", "8",
               "-crf", str(cq), "-pix_fmt", "yuv420p", "-an", output_path]
    elif codec == "libx265":
        cmd = ["ffmpeg", "-y", "-i", clip_path, "-c:v", "libx265", "-preset", "fast",
               "-crf", str(cq), "-pix_fmt", "yuv420p", "-an", output_path]
    else:
        cmd = ["ffmpeg", "-y", "-i", clip_path, "-c:v", codec, "-preset", "fast",
               "-crf", str(cq), "-pix_fmt", "yuv420p", "-an", output_path]

    result = subprocess.run(cmd, capture_output=True, timeout=60)
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


def _get_output_bitrate(output_path: str) -> float:
    """Get average bitrate in Mbps. Returns -1.0 on error."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", "format=bit_rate",
            "-of", "csv=p=0", output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip()) / 1_000_000
    except Exception:
        pass
    return -1.0


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
    Encode to meet VMAF threshold using data-driven CQ selection.

    Single-shot encode at an empirically-chosen CQ, with one VMAF check.
    If below threshold, one rescue re-encode at a lower CQ.
    Typical total time: 35-50s (vs 85-120s with binary search).

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

    _SCENE_TO_ACTION = {
        'screen content / text': 'low-action',
        'animation / cartoon / rendered graphics': 'animation',
        'faces / people': 'low-action',
        'gaming content': 'high-action',
        'other': 'medium-action',
        'unclear': 'default',
        'default': 'default',
    }

    # Conservative CQ values derived from 15 days of production convergence data.
    # Safety buffer: -4 for T93, -3 for T89, -2 for T85 (larger buffer at higher
    # thresholds because the scoring soft-zone penalty is devastating).
    _CONSERVATIVE_CQ = {
        'high': {
            'animation': 29, 'low-action': 37, 'medium-action': 33,
            'high-action': 34, 'default': 34,
        },
        'medium': {
            'animation': 40, 'low-action': 43, 'medium-action': 41,
            'high-action': 38, 'default': 41,
        },
        'low': {
            'animation': 46, 'low-action': 45, 'medium-action': 44,
            'high-action': 42, 'default': 44,
        },
    }

    _RESCUE_CQ_DROP = {'high': 5, 'medium': 4, 'low': 3}

    threshold_int = int(vmaf_threshold)
    quality_tier = 'high' if threshold_int >= 93 else 'medium' if threshold_int >= 88 else 'low'

    if scene_type:
        action_key = _SCENE_TO_ACTION.get(scene_type.lower(), 'default')
        rate = _CONSERVATIVE_CQ[quality_tier].get(action_key, _CONSERVATIVE_CQ[quality_tier]['default'])
        print(f"[iterative] Content-aware CQ: scene='{scene_type}' → "
              f"action='{action_key}' tier='{quality_tier}' → rate={rate}")
    else:
        rate = _CONSERVATIVE_CQ[quality_tier]['default']
        print(f"[iterative] Default CQ (no scene type): tier='{quality_tier}' rate={rate}")

    rate_low = max(rate - 15, AV1_CQ_MIN)
    n_subsample = VMAF_SUBSAMPLE.get(threshold_int, 30)
    hard_cutoff = vmaf_threshold - 5.0

    print(f"[iterative] VMAF: threshold={vmaf_threshold} n_subsample={n_subsample} "
          f"budget={time_budget:.0f}s")

    if codec.endswith('_nvenc'):
        fast_preset = 'p5'
        quality_preset = 'p7'
    elif codec == 'libsvtav1':
        fast_preset = '8'
        quality_preset = '6'
    else:
        fast_preset = 'veryfast'
        quality_preset = 'medium'

    iteration = 0
    temp_files = []

    try:
        # === PHASE 1: First encode at conservative CQ ===
        iteration = 1
        temp_output = f"{output_path}.iter1.mp4"
        temp_files.append(temp_output)

        print(f"[iterative] Encode 1: rate={rate} preset={fast_preset} codec={codec}")
        encode_result = encode_video(
            input_path, temp_output, codec,
            rate=rate, preset=fast_preset,
            scene_type=scene_type, contrast_value=contrast_value,
            codec_mode=codec_mode, target_bitrate=target_bitrate,
            logging_enabled=False
        )

        if encode_result is None or encode_result == (None, None):
            print(f"[iterative] First encode failed at rate={rate}")
            temp_output = None
        elif not os.path.exists(temp_output) or os.path.getsize(temp_output) == 0:
            print(f"[iterative] First encode produced empty output")
            temp_output = None

        # VBR/CBR bitrate enforcement — retry with aggressive CQ bumps
        is_vbr = codec_mode.upper() in ("VBR", "CBR")
        if temp_output and is_vbr:
            max_allowed = target_bitrate * 1.10
            for vbr_retry in range(4):
                actual_mbps = _get_output_bitrate(temp_output)
                if actual_mbps <= 0 or actual_mbps <= max_allowed:
                    if vbr_retry > 0:
                        print(f"[iterative] VBR: compliant at {actual_mbps:.1f}Mbps "
                              f"(limit {max_allowed:.1f}Mbps) after {vbr_retry} retries")
                    break

                if vbr_retry >= 3:
                    print(f"[iterative] VBR: still {actual_mbps:.1f}Mbps after 3 retries, giving up")
                    os.unlink(temp_output)
                    if temp_output in temp_files:
                        temp_files.remove(temp_output)
                    temp_output = None
                    break

                overshoot = actual_mbps / max_allowed
                cq_bump = max(3, int(math.log2(max(overshoot, 1.01)) * 8))
                rate = min(rate + cq_bump, AV1_CQ_MAX)
                print(f"[iterative] VBR: {actual_mbps:.1f}Mbps > {max_allowed:.1f}Mbps, "
                      f"CQ→{rate} (+{cq_bump}, retry {vbr_retry+1}/3)")

                os.unlink(temp_output)
                if temp_output in temp_files:
                    temp_files.remove(temp_output)

                elapsed = time.time() - start_time
                if time_budget - elapsed < 30:
                    print(f"[iterative] VBR: no time for retry ({time_budget - elapsed:.0f}s left)")
                    temp_output = None
                    break

                temp_output = f"{output_path}.vbr{vbr_retry}.mp4"
                temp_files.append(temp_output)
                encode_result = encode_video(
                    input_path, temp_output, codec,
                    rate=rate, preset=fast_preset,
                    scene_type=scene_type, contrast_value=contrast_value,
                    codec_mode=codec_mode, target_bitrate=target_bitrate,
                    logging_enabled=False
                )
                if (encode_result is None or encode_result == (None, None)
                        or not os.path.exists(temp_output)
                        or os.path.getsize(temp_output) == 0):
                    temp_output = None
                    break

        # === PHASE 2: VMAF measurement ===
        vmaf = -1.0
        if temp_output:
            elapsed = time.time() - start_time
            remaining = time_budget - elapsed
            if remaining > SAFETY_MARGIN_SECONDS + 15:
                vmaf = measure_vmaf_fast(input_path, temp_output, n_subsample=n_subsample)

        if temp_output and vmaf >= 0:
            output_size = os.path.getsize(temp_output)
            compression_ratio = input_size / output_size if output_size > 0 else 1.0
            compression_rate = output_size / input_size if input_size > 0 else 1.0

            print(f"[iterative]   → VMAF={vmaf:.2f} ratio={compression_ratio:.1f}x "
                  f"size={output_size / 1024 / 1024:.1f}MB")

            # === PHASE 3: Decision ===
            if vmaf >= vmaf_threshold:
                # ABOVE threshold — check if we have time for a quality upgrade
                elapsed = time.time() - start_time
                remaining = time_budget - elapsed

                if remaining > 35:
                    upgrade_output = f"{output_path}.upgrade.mp4"
                    temp_files.append(upgrade_output)
                    print(f"[iterative] Quality upgrade: re-encoding at {quality_preset} "
                          f"rate={rate} ({remaining:.0f}s remaining)")
                    upgrade_result = encode_video(
                        input_path, upgrade_output, codec,
                        rate=rate, preset=quality_preset,
                        scene_type=scene_type, contrast_value=contrast_value,
                        codec_mode=codec_mode, target_bitrate=target_bitrate,
                        logging_enabled=False
                    )

                    if (upgrade_result is not None and upgrade_result != (None, None)
                            and os.path.exists(upgrade_output)
                            and os.path.getsize(upgrade_output) > 0):
                        if is_vbr:
                            upgrade_mbps = _get_output_bitrate(upgrade_output)
                            if upgrade_mbps > 0 and upgrade_mbps > target_bitrate * 1.10:
                                print(f"[iterative] Upgrade exceeds VBR limit "
                                      f"({upgrade_mbps:.1f}Mbps), keeping original")
                                os.unlink(upgrade_output)
                                if upgrade_output in temp_files:
                                    temp_files.remove(upgrade_output)
                                upgrade_result = None

                    if (upgrade_result is not None and upgrade_result != (None, None)
                            and os.path.exists(upgrade_output)
                            and os.path.getsize(upgrade_output) > 0):
                        upgrade_size = os.path.getsize(upgrade_output)
                        upgrade_ratio = input_size / upgrade_size
                        upgrade_rate = upgrade_size / input_size

                        elapsed2 = time.time() - start_time
                        remaining2 = time_budget - elapsed2
                        if remaining2 > 20:
                            upgrade_vmaf = measure_vmaf_fast(
                                input_path, upgrade_output, n_subsample=n_subsample)
                            if upgrade_vmaf >= vmaf_threshold:
                                print(f"[iterative] Upgrade: VMAF={upgrade_vmaf:.2f} "
                                      f"ratio={upgrade_ratio:.1f}x (was {compression_ratio:.1f}x)")
                                os.unlink(temp_output)
                                if temp_output in temp_files:
                                    temp_files.remove(temp_output)
                                temp_output = upgrade_output
                                vmaf = upgrade_vmaf
                                compression_ratio = upgrade_ratio
                                compression_rate = upgrade_rate
                                iteration = 2
                            else:
                                print(f"[iterative] Upgrade below threshold ({upgrade_vmaf:.2f}), keeping original")
                                os.unlink(upgrade_output)
                                if upgrade_output in temp_files:
                                    temp_files.remove(upgrade_output)
                        else:
                            print(f"[iterative] Upgrade accepted without VMAF verify "
                                  f"(ratio={upgrade_ratio:.1f}x)")
                            os.unlink(temp_output)
                            if temp_output in temp_files:
                                temp_files.remove(temp_output)
                            temp_output = upgrade_output
                            compression_ratio = upgrade_ratio
                            compression_rate = upgrade_rate
                            iteration = 2
                    else:
                        if os.path.exists(upgrade_output):
                            os.unlink(upgrade_output)
                        if upgrade_output in temp_files:
                            temp_files.remove(upgrade_output)

                shutil.move(temp_output, output_path)
                if temp_output in temp_files:
                    temp_files.remove(temp_output)
                total_time = time.time() - start_time
                print(f"[iterative] Success: VMAF={vmaf:.2f} "
                      f"ratio={compression_ratio:.1f}x "
                      f"rate={rate} iterations={iteration} "
                      f"time={total_time:.1f}s")
                return {
                    'success': True,
                    'vmaf': vmaf,
                    'rate': rate,
                    'compression_ratio': compression_ratio,
                    'compression_rate': compression_rate,
                    'iterations': iteration,
                    'time': total_time,
                    'fallback': False,
                }

            else:
                # BELOW threshold
                if is_vbr:
                    # VBR mode: accept suboptimal VMAF to preserve bitrate compliance.
                    # Rescue lowers CQ → raises bitrate → instant zero from validator.
                    shutil.move(temp_output, output_path)
                    if temp_output in temp_files:
                        temp_files.remove(temp_output)
                    total_time = time.time() - start_time
                    print(f"[iterative] VBR accept: VMAF={vmaf:.2f} below {vmaf_threshold} "
                          f"but bitrate-compliant (rescue would blow bitrate)")
                    return {
                        'success': True,
                        'vmaf': vmaf,
                        'rate': rate,
                        'compression_ratio': compression_ratio,
                        'compression_rate': compression_rate,
                        'iterations': iteration,
                        'time': total_time,
                        'fallback': False,
                    }

                # CRF mode: rescue re-encode to boost VMAF
                os.unlink(temp_output)
                if temp_output in temp_files:
                    temp_files.remove(temp_output)

                if vmaf >= hard_cutoff:
                    rescue_drop = _RESCUE_CQ_DROP.get(quality_tier, 4)
                    rescue_rate = max(rate - rescue_drop, AV1_CQ_MIN)
                    print(f"[iterative] Soft zone (VMAF={vmaf:.2f}), rescue at rate={rescue_rate} "
                          f"(dropped {rescue_drop} from {rate})")
                else:
                    rescue_rate = rate_low
                    print(f"[iterative] Hard fail (VMAF={vmaf:.2f} < {hard_cutoff}), "
                          f"rescue at rate_low={rate_low}")

                iteration = 2
                rescue_output = f"{output_path}.rescue.mp4"
                temp_files.append(rescue_output)
                encode_video(
                    input_path, rescue_output, codec,
                    rate=rescue_rate, preset=fast_preset,
                    scene_type=scene_type, contrast_value=contrast_value,
                    codec_mode=codec_mode, target_bitrate=target_bitrate,
                    logging_enabled=False
                )

                if os.path.exists(rescue_output) and os.path.getsize(rescue_output) > 0:
                    rescue_size = os.path.getsize(rescue_output)
                    shutil.move(rescue_output, output_path)
                    if rescue_output in temp_files:
                        temp_files.remove(rescue_output)
                    total_time = time.time() - start_time
                    rescue_ratio = input_size / rescue_size
                    rescue_rate_val = rescue_size / input_size
                    print(f"[iterative] Success (rescue): ratio={rescue_ratio:.1f}x "
                          f"rate={rescue_rate} iterations={iteration} "
                          f"time={total_time:.1f}s")
                    return {
                        'success': True,
                        'vmaf': None,
                        'rate': rescue_rate,
                        'compression_ratio': rescue_ratio,
                        'compression_rate': rescue_rate_val,
                        'iterations': iteration,
                        'time': total_time,
                        'fallback': False,
                    }

        # === FALLBACK: everything failed, encode at most conservative rate ===
        print(f"[iterative] All attempts failed, fallback at rate={rate_low}")
        encode_video(
            input_path, output_path, codec,
            rate=rate_low,
            scene_type=scene_type, contrast_value=contrast_value,
            codec_mode=codec_mode, target_bitrate=target_bitrate,
            logging_enabled=False
        )

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
