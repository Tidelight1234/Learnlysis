# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""Phase 2 抽帧：按时间点从视频文件抓关键帧，转成多模态 ``data:`` URI 喂给 LLM。

**关键约束（见 video_player.py L109-114）**：绝不在直播 ``QMediaPlayer`` 上挂
``QVideoFrame`` / ``VideoSurface``——那个 flag 要求后端提供 ``QVideoSurfaceControl``，而
Windows 的 WMF 后端只给 ``QVideoWindowControl``，会导致 Qt 跳过 WMF、回退 DirectShow，进而
让 MP4 因缺解码器放不了。所以本模块走一条**与直播播放器完全无关的独立离线解码通道**：
直接对视频**文件**抽帧，优先 ``opencv-python``（``cv2``），其次系统 ``ffmpeg`` CLI。二者都是
可选依赖（未随核心安装），缺席时如实降级并给出安装命令，绝不静默失败、也绝不去动直播播放器。

音频：本阶段不做重活（ASR 属 Phase 3 的 faster-whisper）。话语转写由用户在 auto_coding 的
输入框里粘贴，或交给能直接吃音频的多模态模型；``video_probe`` 只负责画面帧。

抽帧后按最长边缩放再转 JPEG base64，控制多模态 prompt 体积（BYO key = 用户自付，成本敏感）。
"""

from __future__ import annotations

import base64
import importlib.util
import logging
import os
import shutil
import subprocess

from PySide6.QtCore import QBuffer, Qt
from PySide6.QtGui import QImage

#: 抽帧后按最长边缩放到此像素以内（够多模态模型看清画面，又不至于把 prompt 撑爆）
MAX_SIDE = 1280
#: JPEG 质量：画质与体积的折中
JPEG_QUALITY = 82
#: 单帧 ffmpeg 解码超时（秒）
FFMPEG_TIMEOUT = 30


# ------------------------------------------------------------------ 后端探测

def optional_backends():
    """可选抽帧后端在场判定（照 analysis_ml / ai_client 的 ``find_spec`` 范式）。

    返回 ``{"opencv": bool, "ffmpeg": bool}``。两者缺席不影响软件其余功能，只让 Phase 2
    的「视频抽帧」这一条路降级为不可用并写明原因。
    """
    return {
        "opencv": importlib.util.find_spec("cv2") is not None,
        "ffmpeg": shutil.which("ffmpeg") is not None,
    }


def frames_available():
    """``(是否至少有一个抽帧后端, 后端表)``。"""
    backends = optional_backends()
    return any(backends.values()), backends


def install_hint():
    """没有抽帧后端时给用户的人话 + 安装命令（照 torch 的可选后端降级口径）。"""
    return ("视频抽帧需要 opencv-python 或系统 ffmpeg（均为可选依赖，未随核心安装）：\n"
            "  · pip install opencv-python\n"
            "  · 或安装 ffmpeg 并确保在 PATH 中（命令行 ffmpeg -version 可验证）\n"
            "装好后重开本页即可用「AI 辅助编码」抽取关键帧。")


# ------------------------------------------------------------------ 各后端抽帧

def _bgr_to_qimage(frame):
    """cv2 的 BGR ``HxWx3`` uint8 数组 → QImage（深拷贝，脱离 numpy 缓冲生命周期）。"""
    height, width = frame.shape[0], frame.shape[1]
    rgb = frame[:, :, ::-1].copy()               # BGR → RGB，copy 保证连续内存
    image = QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888)
    return image.copy()


def _extract_opencv(path, seconds):
    """opencv 后端：按毫秒定位读一帧。返回 ``(QImage or None, note)``。"""
    import cv2                                   # noqa: 延迟导入，缺席时 optional_backends 已挡
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        return None, "opencv 打不开该视频文件（路径无效或编码不受支持）"
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(seconds)) * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None:
            return None, "opencv 在该时间点没读到帧（可能超出视频时长或该帧不可解码）"
        return _bgr_to_qimage(frame), ""
    finally:
        capture.release()


def _extract_ffmpeg(path, seconds):
    """ffmpeg 后端：精确 seek（``-ss`` 放在 ``-i`` 之后）取一帧 PNG。返回 ``(QImage or None, note)``。"""
    exe = shutil.which("ffmpeg")
    if not exe:
        return None, "未找到 ffmpeg 可执行文件"
    # -ss 放 -i 之后是「输出定位」，逐帧解到目标时刻，准确但对长视频偏慢；课堂片段可接受
    command = [exe, "-nostdin", "-loglevel", "error", "-i", path,
               "-ss", "%.3f" % max(0.0, float(seconds)),
               "-frames:v", "1", "-f", "image2", "-c:v", "png", "-"]
    try:
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=FFMPEG_TIMEOUT)
    except subprocess.TimeoutExpired:
        return None, "ffmpeg 抽帧超时（>%d 秒）" % FFMPEG_TIMEOUT
    except Exception as exc:                     # noqa: BLE001 调用失败要给人话
        return None, "ffmpeg 调用失败：%s" % exc
    data = proc.stdout or b""
    if not data:
        detail = (proc.stderr or b"").decode("utf-8", "replace").strip()[:200]
        return None, "ffmpeg 未产出帧：%s" % (detail or "该时间点无输出")
    image = QImage.fromData(data)
    if image.isNull():
        return None, "ffmpeg 产出的字节无法解析为图像"
    return image, ""


_BACKENDS = {"opencv": _extract_opencv, "ffmpeg": _extract_ffmpeg}


def extract_frame(path, seconds, backend="auto"):
    """在 ``seconds`` 处抽一帧。返回 ``(QImage or None, note)``。

    ``backend`` 为 ``"auto"``（按 opencv→ffmpeg 顺序试在场的那个）或指定 ``"opencv"`` /
    ``"ffmpeg"``。文件不存在、无后端、抽不到帧都返回 ``(None, 人话原因)``，绝不抛给上层。
    """
    if not path or not os.path.exists(path):
        return None, "视频文件不存在：%s" % (path or "（未加载视频）")
    present = optional_backends()
    if backend in _BACKENDS:
        order = [backend]
    else:
        order = [name for name in ("opencv", "ffmpeg") if present.get(name)]
    if not order:
        return None, install_hint()
    last_note = ""
    for name in order:
        if name not in _BACKENDS:
            continue
        try:
            image, note = _BACKENDS[name](path, seconds)
        except Exception as exc:                 # noqa: BLE001 单个后端崩不影响另一个
            image, note = None, "%s 抽帧异常：%s" % (name, exc)
            logging.warning("video_probe %s 抽帧失败：%s", name, exc)
        if image is not None and not image.isNull():
            return image, ""
        last_note = note or last_note
    return None, last_note or "抽帧失败（未知原因）"


# ------------------------------------------------------------------ 图像 → data URI

def qimage_to_data_uri(image, max_side=MAX_SIDE, quality=JPEG_QUALITY, fmt="JPEG"):
    """QImage → ``data:image/jpeg;base64,...``（按最长边缩放后编码）。空图返回空串。"""
    if image is None or image.isNull():
        return ""
    picture = image
    if max_side and max(picture.width(), picture.height()) > max_side:
        picture = picture.scaled(max_side, max_side, Qt.KeepAspectRatio,
                                 Qt.SmoothTransformation)
    buffer = QBuffer()
    buffer.open(QBuffer.WriteOnly)
    try:
        picture.save(buffer, fmt, int(quality))
        raw = bytes(buffer.data())
    finally:
        buffer.close()
    mime = "image/png" if str(fmt).upper() == "PNG" else "image/jpeg"
    return "data:%s;base64,%s" % (mime, base64.b64encode(raw).decode("ascii"))


def extract_frame_uri(path, seconds, **kwargs):
    """抽一帧并直接转 data URI。返回 ``(uri or "", note)``。"""
    image, note = extract_frame(path, seconds)
    if image is None:
        return "", note
    return qimage_to_data_uri(image, **kwargs), ""


def extract_frames(path, seconds_list, **kwargs):
    """批量抽帧：返回 ``[(seconds, uri or "", note), ...]``，与传入时间点一一对应。"""
    out = []
    for seconds in seconds_list or []:
        uri, note = extract_frame_uri(path, seconds, **kwargs)
        out.append((seconds, uri, note))
    return out
