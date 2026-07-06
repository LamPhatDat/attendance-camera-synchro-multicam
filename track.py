import cv2
import zmq
import threading
import os
import time
import numpy as np
from ultralytics import YOLO

from gallery import ReIDGallery
from tracker import CameraTracker
import zeroconf_utils


_context = zmq.Context()
_pull = _context.socket(zmq.PULL)


def _resolve_zmq_endpoint():
    info = zeroconf_utils.discover_service("_gateway-zmq._tcp", timeout=1.0)
    if info:
        _, ip, port = zeroconf_utils.resolve_service(info)
        return f"tcp://{ip}:{port}"
    return os.environ.get("ZMQ_ENDPOINT", "tcp://127.0.0.1:5557")


def _zmq_listener(gallery):
    current_endpoint = None
    while True:
        if current_endpoint is None:
            endpoint = _resolve_zmq_endpoint()
            if endpoint:
                _pull.connect(endpoint)
                current_endpoint = endpoint
                print(f"[TRACK] Connected to ZMQ at {endpoint}")
            else:
                time.sleep(2)
                continue

        try:
            msg = _pull.recv_json()
            if msg["action"] == "track":
                gallery.add_pending_target(msg["name"])
                print(f"[TRACK] Target added: {msg['name']} — queue size: {len(gallery.get_pending_targets_snapshot())}")
        except zmq.ZMQError as e:
            print(f"[TRACK] ZMQ connection lost: {e} — reconnecting...")
            current_endpoint = None
            time.sleep(1)
        except Exception as e:
            print(f"[TRACK] ZMQ error: {e}")


def main():
    gallery = ReIDGallery(
        reid_weights="osnet_x1_0_msmt17.pth",
    )

    threading.Thread(target=_zmq_listener, args=(gallery,), daemon=True).start()

    rtsp_url = "rtsp://admin:L26DDDDF@10.40.90.225:554/cam/realmonitor?channel=1&subtype=1"

    roi = (400, 50, 560, 350)
    door_poly = np.array([(430, 0), (530, 0), (510, 230), (430, 160)], dtype=np.int32)

    yolo = YOLO("yolo11s.pt", task="detect")
    cam = CameraTracker(
        cam_id="cam1",
        source_url=rtsp_url,
        gallery=gallery,
        yolo_model=yolo,
        roi=roi,
        door_poly=door_poly,
    )

    for frame, detections in cam.track_loop():

        # --- FUTURE HOMOGRAPHY STEP ---
        for person in detections:
            track_id = person.get("global_id", person["id"])
            feet_coords = person["base_point"]
        # ------------------------------

        small_frame = cv2.resize(frame, (480, 384))

        linked = gallery.get_linked_targets_snapshot()
        pending = gallery.get_pending_targets_snapshot()
        if linked:
            names = list(linked.values())
            label = f"TRACKING: {', '.join(names)}"
            color = (0, 255, 0)
            if pending:
                label += f" (+{len(pending)} waiting)"
        elif pending:
            label = f"WAITING: {len(pending)} target(s) for ROI"
            color = (255, 165, 0)
        else:
            label = "No target set"
            color = (100, 100, 100)
        cv2.putText(small_frame, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("OSNet ReID Tracking", small_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
