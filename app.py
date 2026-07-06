import logging
import os
import threading
import time
import subprocess
import cv2
import psutil
import numpy as np
from collections import deque
from flask import Flask, jsonify, render_template, Response
from ultralytics import YOLO

from gallery import ReIDGallery
from tracker import CameraTracker
import zeroconf_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

gallery: ReIDGallery | None = None
cameras: dict[str, CameraTracker] = {}
camera_buffers: dict[str, deque] = {}
latest_dets: dict[str, list[dict]] = {}
latest_frame_idx: dict[str, int] = {}
lock = threading.Lock()

_proc = psutil.Process()
_proc.cpu_percent()


def _get_gpu_percent() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        vals = [float(v.strip()) for v in out.stdout.strip().split('\n') if v.strip()]
        return vals[0] if vals else 0.0
    except Exception:
        return 0.0


def _get_cpu_percent() -> float:
    return _proc.cpu_percent(interval=0) / psutil.cpu_count()


def _resolve_zmq_endpoint():
    info = zeroconf_utils.discover_service("_gateway-zmq._tcp", timeout=2.0)
    if info:
        _, ip, port = zeroconf_utils.resolve_service(info)
        return f"tcp://{ip}:{port}"
    return os.environ.get("ZMQ_ENDPOINT", "tcp://10.40.90.214:5557")


def _zmq_listener():
    import zmq

    global gallery

    context = zmq.Context()
    pull = context.socket(zmq.PULL)
    current_endpoint = None

    # Wait until gallery is created
    while gallery is None:
        time.sleep(0.5)

    while True:
        if current_endpoint is None:
            endpoint = _resolve_zmq_endpoint()
            if endpoint:
                pull.connect(endpoint)
                current_endpoint = endpoint
                logger.info("[ZMQ] Connected to gateway at %s", endpoint)
            else:
                time.sleep(2)
                continue

        try:
            msg = pull.recv_json()
            if msg["action"] == "track":
                gallery.add_pending_target(msg["name"])
                print(f"[TRACK SIGNAL] name={msg['name']} — queue_size={len(gallery.get_pending_targets_snapshot())}")
        except zmq.ZMQError as e:
            logger.warning("[ZMQ] Connection lost: %s — reconnecting...", e)
            current_endpoint = None
            time.sleep(1)
        except Exception as e:
            logger.error("[FLASK] ZMQ error: %s", e)


def _track_worker():
    global gallery, latest_dets, latest_frame_idx

    gallery = ReIDGallery(reid_weights="osnet_x1_0_msmt17.pth")

    # ── Camera configs (add more entries for multi-camera) ──
    camera_configs = [
        {
            "cam_id": "cam1",
            "source_url": os.environ.get(
                "RTSP_URL",
                "rtsp://admin:HungVuong@2023!@10.40.20.38:554/cam/realmonitor?channel=1&subtype=1"
                # "rtsp://admin:L26DDDDF@10.40.90.235:554/cam/realmonitor?channel=1&subtype=1",
              ),
        #     # "roi": (440, 278, 730, 502),
        #     # "door_poly": np.array([(1124, 54), (1136, 348), (1240, 314), (1248, 48)], dtype=np.int32),
            "roi": (24, 68, 92, 136),
            "door_poly": np.array([(834, 54), (870, 292), (998, 234), (968, 48)], dtype=np.int32),
        },
        # {
        #     "cam_id": "cam2",
        #     "source_url": os.environ.get(
        #         "RTSP_URL2",
        #         "rtsp://admin:HungVuong@2023!@10.40.20.61:554/cam/realmonitor?channel=1&subtype=1",
        #     ),
        #     # "roi": (676, 384, 954, 760),
        #     # "door_poly": np.array([(1118, 46), (1116, 354), (1248, 308), (1248, 50)], dtype=np.int32), 
        #     # "roi": (1890, 50, 2270, 620),
        #     # "door_poly": np.array([(1112, 48), (1134, 340), (1252, 316), (1250, 54)], dtype=np.int32),
        #     # "roi": (1480, 50, 1754, 500),
        #     # "door_poly": np.array([(1118, 52), (1124, 356), (1250, 324), (1246, 42)], dtype=np.int32),
        #     "roi": (676, 150, 910, 428),
        #     "door_poly": np.array([(1990, 86), (2000, 264), (2106, 228), (2098, 50)], dtype=np.int32),
        # },
        {
            "cam_id": "cam3",
            "source_url": os.environ.get(
                "RTSP_URL3",
                "rtsp://admin:L26DDDDF@10.40.90.235:554/cam/realmonitor?channel=1&subtype=1",
            ),
            # "roi": (440, 278, 730, 502),
            # "door_poly": np.array([(1124, 54), (1136, 348), (1240, 314), (1248, 48)], dtype=np.int32),
            "roi": (463, 47, 624, 407),
            "door_poly": np.array([(102, 190), (119, 330), (158, 326), (156, 186)], dtype=np.int32),
        },  

        # {
        #     "cam_id": "cam4",
        #     "source_url": "rtsp://admin:HungVuong@2023!@10.40.20.21:554/cam/realmonitor?channel=1&subtype=1",
        #     "roi": (440, 278, 730, 502),
        #     "door_poly": np.array([(1124, 54), (1136, 348), (1240, 314), (1248, 48)], dtype=np.int32),
        # },
        # {
        #     "cam_id": "cam5",
        #     "source_url": "rtsp://admin:HungVuong@2023!@10.40.20.23:554/cam/realmonitor?channel=1&subtype=1",
        #     "roi": (440, 278, 730, 502),
        #     "door_poly": np.array([(1124, 54), (1136, 348), (1240, 314), (1248, 48)], dtype=np.int32),
        # },
    ]


    for cfg in camera_configs:
        yolo = YOLO("yolo11s.pt", task="detect")
        cam = CameraTracker(
            cam_id=cfg["cam_id"],
            source_url=cfg["source_url"],
            gallery=gallery,
            yolo_model=yolo,
            roi=cfg["roi"],
            door_poly=cfg["door_poly"],
            imgsz=cfg.get("imgsz", 640),
        )
        cameras[cfg["cam_id"]] = cam
        camera_buffers[cfg["cam_id"]] = deque(maxlen=1)
        logger.info("Camera '%s' configured: %s", cfg["cam_id"], cfg["source_url"])

    def _run_camera(cam: CameraTracker):
        for frame, detections in cam.track_loop():
            ret, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                continue
            camera_buffers[cam.cam_id].append(jpeg.tobytes())   # ← ra ngoài khoá
            with lock:
                latest_dets[cam.cam_id] = detections
                latest_frame_idx[cam.cam_id] = cam.frame_idx

    for cam_id, cam in cameras.items():
        if cam_id != "cam1":
            t = threading.Thread(target=_run_camera, args=(cam,), daemon=True)
            t.start()
            logger.info("Camera '%s' started in background thread", cam_id)

    # Keep cam1 in this thread (blocking)
    _run_camera(cameras["cam1"])

# 8h29 - 29/06
def _generate_frames(cam_id: str = "cam1"):
    last = None
    while True:
        buf = camera_buffers.get(cam_id)
        frame_bytes = buf[-1] if buf else None
        if frame_bytes is not None and frame_bytes is not last:
            last = frame_bytes
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        else:
            time.sleep(0.03)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
@app.route("/video_feed/<cam_id>")
def video_feed(cam_id="cam1"):
    return Response(
        _generate_frames(cam_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
@app.route("/status/<cam_id>")
def status(cam_id="cam1"):
    with lock:
        dets = latest_dets.get(cam_id, [])
        frame_gids = [d["global_id"] for d in dets]

        linked_snapshot = gallery.get_linked_targets_snapshot() if gallery else {}
        pending_list = gallery.get_pending_targets_snapshot() if gallery else []

        linked_in_frame = sum(1 for gid in frame_gids if gid in linked_snapshot)
        linked_in_frame_names = [d["name"] for d in dets if d["global_id"] in linked_snapshot]

        known_list = []
        for gid, name in linked_snapshot.items():
            slot_count = 0
            if gallery is not None:
                gid_neg = next((sid for sid, n in gallery.slot_faiss_id_to_name.items() if n == name), None)
                if gid_neg is not None:
                    slot_count = 1
            known_list.append({"name": name, "global_id": gid, "slot_count": slot_count})

        unknown_list = []
        if gallery is not None:
            emb_count = gallery.get_embedding_count()
            total_tracked = sum(len(cam.trackid_to_global) for cam in cameras.values())
        else:
            emb_count = 0
            total_tracked = 0

        return jsonify(
            linked_targets={str(gid): name for gid, name in linked_snapshot.items()},
            pending_targets=pending_list,
            detection_count=len(dets),
            total_tracked=total_tracked,
            linked_in_frame=linked_in_frame,
            linked_in_frame_names=linked_in_frame_names,
            embedding_known=known_list,
            embedding_unknown=[gid for gid in range(emb_count) if gid not in linked_snapshot],
            cpu_percent=_get_cpu_percent(),
            gpu_percent=_get_gpu_percent(),
        )


@app.route("/embeddings")
def embeddings_status():
    if gallery is None:
        return jsonify({"error": "gallery not ready"})

    linked = gallery.get_linked_targets_snapshot()
    return jsonify({
        "index_ntotal": gallery.get_index_ntotal(),
        "linked_targets": {str(gid): name for gid, name in linked.items()},
        "pending_targets": gallery.get_pending_targets_snapshot(),
    })
@app.route("/boxes")
@app.route("/boxes/<cam_id>")
def boxes(cam_id="cam1"):
    """Trả tọa độ bounding box của mọi người đang track ở 1 camera."""
    with lock:
        dets = latest_dets.get(cam_id, [])
        result = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            result.append({
                "id": d["id"],                 # track_id của ByteTrack
                "global_id": d["global_id"],   # ID xuyên camera
                "name": d["name"],             # tên hoặc "Unknown (ID n)"
                "bbox": [x1, y1, x2, y2],       # góc trái-trên, phải-dưới
                "center": [(x1 + x2) // 2, (y1 + y2) // 2],  # tâm box
                "foot": d["base_point"],       # điểm chân (x giữa, y đáy)
            })
    return jsonify({
        "cam_id": cam_id,
        "count": len(result),
        "boxes": result,
    })

if __name__ == "__main__":
    _zc_camera = zeroconf_utils.advertise_service(
        "_camera-http._tcp", "CameraHTTP", 5150
    )
    logger.info("Zeroconf: advertising _camera-http._tcp (port 5150)")

    threading.Thread(target=_zmq_listener, daemon=True).start()
    threading.Thread(target=_track_worker, daemon=True).start()

    app.run(host="0.0.0.0", port=5150, threaded=True, debug=False)
