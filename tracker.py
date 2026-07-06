import logging
import cv2
import numpy as np

from gallery import ReIDGallery
from face_detection import FaceDetector
from arcface_recognizer import ArcFaceRecognizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraTracker:
    def __init__(
        self,
        cam_id: str,
        source_url: str,
        gallery: ReIDGallery,
        yolo_model,
        roi: tuple[int, int, int, int],
        door_poly: np.ndarray,
        tracker_config: str = "custom_tracker.yaml",
        occlusion_iou_threshold: float = 0.5,
        stability_frames: int = 3,
        track_ttl: int = 150,
        roi_entry_stability: int = 3,
        imgsz: int = 640,
        conf: float = 0.15,
        margin_threshold = 0.06,
    ):
        self.cam_id = cam_id
        self.source_url = source_url
        self.gallery = gallery
        self.model = yolo_model
        self.tracker_config = tracker_config
        self.ROI = roi
        self.DOOR_POLYGON = door_poly

        self.OCCLUSION_IOU_THRESHOLD = occlusion_iou_threshold
        self.STABILITY_FRAMES = stability_frames
        self.TRACK_TTL = track_ttl
        self.ROI_ENTRY_STABILITY = roi_entry_stability
        self.IMGSZ = imgsz
        self.CONF = conf

        self.trackid_to_global: dict[int, int] = {}
        self.track_history: dict[int, int] = {}
        self.track_last_seen: dict[int, int] = {}
        self.track_last_position: dict[int, tuple[int, int]] = {}
        self.roi_entry_frames: dict[int, int] = {}
        self.frame_idx = 0
        self.face_detector = None
        self.face_recognizer = None
        self.last_face_try: dict[int, int] = {}
        if self.cam_id in ["cam3"]:
            try:
                self.face_detector = FaceDetector()
                self.face_recognizer = ArcFaceRecognizer()
                self.face_recognizer.load_database("data")
                logging.info("[%s] Face tie-breaker BAT", self.cam_id)
            except Exception as e:
                logging.warning("[%s] Khong nap duoc face: %s", self.cam_id, e)

        self.COLOR_PALETTE = np.array([
            (255, 50, 50), (50, 255, 50), (50, 50, 255),
            (255, 255, 50), (50, 255, 255), (255, 50, 255),
            (255, 150, 50), (150, 50, 255), (50, 150, 255), (150, 255, 50),
        ], dtype=np.uint8)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _in_roi(self, point: tuple[int, int]) -> bool:
        x, y = point
        rx1, ry1, rx2, ry2 = self.ROI
        return rx1 <= x <= rx2 and ry1 <= y <= ry2

    def _in_door_area(self, point: tuple[int, int]) -> bool:
        return cv2.pointPolygonTest(self.DOOR_POLYGON, point, False) >= 0

    def get_iou(self, boxA, boxB) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    # ------------------------------------------------------------------
    # Stale track eviction
    # ------------------------------------------------------------------

    def _evict_stale_tracks(self):
        stale = [
            t_id
            for t_id, last in self.track_last_seen.items()
            if self.frame_idx - last > self.TRACK_TTL
        ]
        for t_id in stale:
            self.track_last_position.pop(t_id, None)
            self.trackid_to_global.pop(t_id, None)
            # KHÔNG unlink nữa: giữ tên gắn với global_id để ReID khớp lại là tên còn
            self.track_history.pop(t_id, None)
            self.track_last_seen.pop(t_id, None)

        evicted = self.gallery.evict_unnamed()
        for gid in evicted:
            dead = [t for t, g in self.trackid_to_global.items() if g == gid]
            for t in dead:
                self.trackid_to_global.pop(t, None)
                self.track_history.pop(t, None)
                self.track_last_seen.pop(t, None)
                self.track_last_position.pop(t, None)

    # ------------------------------------------------------------------
    # Main tracking loop
    # ------------------------------------------------------------------

    def track_loop(self):
        while True:
            try:
                results = self.model.track(
                    source=self.source_url,
                    conf=self.CONF,
                    imgsz=self.IMGSZ,
                    vid_stride=2,
                    stream=True,
                    tracker=self.tracker_config,
                    persist=True,
                    classes=[0],
                )

                for r in results:
                    self.frame_idx += 1
                    frame = r.orig_img
                    current_detections = []

                    self._evict_stale_tracks()

                    assigned_in_frame: dict[int, float] = {}

                    if r.boxes is not None and r.boxes.id is not None:
                        data = r.boxes.data.cpu().numpy()
                        all_boxes = data[:, :4]
                        order = np.argsort(-data[:, 5])
                        data = data[order]

                        for row in data:
                            x1, y1, x2, y2 = row[:4]
                            track_id = int(row[4])
                            score = float(row[5])
                            box = [int(x1), int(y1), int(x2), int(y2)]

                            self.track_last_seen[track_id] = self.frame_idx
                            self.track_last_position[track_id] = (int((x1 + x2) / 2), int(y2))
                            global_id = None
                            emb = None

                            if track_id in self.trackid_to_global:
                                global_id = self.trackid_to_global[track_id]
                                emb = self.gallery.get_embedding(frame, box)
                                if emb is not None:
                                    emb /= np.linalg.norm(emb) + 1e-6
                                    self.gallery.update_embedding(global_id, emb)
                                    self.gallery.mark_seen(global_id)
                            else:
                                is_occluded = any(
                                    not np.array_equal(box, other.astype(int))
                                    and self.get_iou(box, other) > self.OCCLUSION_IOU_THRESHOLD
                                    for other in all_boxes
                                )

                                if not is_occluded:
                                    self.track_history[track_id] = (
                                        self.track_history.get(track_id, 0) + 1
                                    )

                                    if self.track_history[track_id] >= self.STABILITY_FRAMES:
                                        emb = self.gallery.get_embedding(frame, box)
                                        if emb is not None:
                                            emb /= np.linalg.norm(emb) + 1e-6
                                            best_gid, was_matched = self.gallery.match_or_register(emb)
                                            if was_matched:
                                                self.gallery.update_embedding(best_gid, emb)
                                                self.gallery.mark_seen(best_gid)
                                            self.trackid_to_global[track_id] = best_gid
                                            self.gallery.mark_seen(best_gid)
                                            global_id = best_gid
                                            # face_name, face_conf = self._face_identify(frame, box)
                                            # if face_name is not None and face_conf >= 0.5:
                                            #     self.gallery.link_in_roi(best_gid, face_name)

                            if global_id is not None:
                                if global_id in assigned_in_frame:
                                    if score <= assigned_in_frame[global_id]:
                                        continue
                                assigned_in_frame[global_id] = score

                                base_point = (int((x1 + x2) / 2), int(y2))
                                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                                # ─── FACE tự đóng dấu (chỉ cam có face, so sánh và gán khuôn mặt) ───
                                if self.face_recognizer is not None:
                                    last = self.last_face_try.get(track_id, -999)
                                    if self.frame_idx - last >= 8:      # thử lại mỗi ~8 frame
                                        self.last_face_try[track_id] = self.frame_idx
                                        face_name, face_conf, face_box = self._face_identify(frame, box)
                                        if face_name is not None and face_conf >= 0.65:
                                            self.gallery.force_link(global_id, face_name)
                                            if emb is not None:
                                                self.gallery.capture_slot(face_name, emb)
                                        if face_box is not None:
                                            fx1, fy1, fx2, fy2 = face_box
                                            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                                            flabel = face_name if face_name else "?"
                                            cv2.putText(frame, flabel, (fx1, fy1 - 6),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                # ─── ROI LINKING ───
                                if not self.gallery.is_linked(global_id):
                                    if self._in_roi(center):
                                        self.roi_entry_frames[global_id] = (
                                            self.roi_entry_frames.get(global_id, 0) + 1
                                        )
                                    else:
                                        self.roi_entry_frames[global_id] = 0

                                    if self.roi_entry_frames.get(global_id, 0) >= self.ROI_ENTRY_STABILITY:
                                        pending_name = self.gallery.peek_pending_target()
                                        if pending_name is not None:
                                            if self.gallery.link_in_roi(global_id, pending_name):
                                                if emb is not None:
                                                    self.gallery.capture_slot(pending_name, emb)
                                                self.roi_entry_frames[global_id] = 0

                                # ─── Pending capture cho target đã link ───
                                if emb is not None:
                                    self.gallery.consume_pending_capture(global_id, emb)

                                display_name = self.gallery.get_linked_name(global_id)
                                if display_name is None:
                                    display_name = f"Unknown (ID {global_id})"

                                current_detections.append({
                                    "id": track_id,
                                    "global_id": global_id,
                                    "bbox": box,
                                    "base_point": base_point,
                                    "name": display_name,
                                })

                                color_idx = global_id % len(self.COLOR_PALETTE)
                                box_color = tuple(int(c) for c in self.COLOR_PALETTE[color_idx])

                                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), box_color, 3)

                                # # Draw face box if detector is initialized
                                # if self.face_detector is not None:
                                #     fx1, fy1, fx2, fy2 = box
                                #     fh_crop = fy2 - fy1
                                #     face_crop = frame[fy1:fy1 + int(fh_crop * 0.4), fx1:fx2]
                                #     if face_crop.size > 0:
                                #         try:
                                #             faces = self.face_detector.detect(face_crop)
                                #             if faces:
                                #                 face = max(faces, key=lambda f: f["w"] * f["h"])
                                #                 face_x1 = fx1 + face["x"]
                                #                 face_y1 = fy1 + face["y"]
                                #                 face_x2 = face_x1 + face["w"]
                                #                 face_y2 = face_y1 + face["h"]
                                #                 cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), (0, 255, 255), 2)
                                #                 cv2.putText(frame, "Face", (face_x1, face_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                                #         except Exception:
                                #             pass

                                label = f"TRACKING: {display_name}"
                                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                cv2.rectangle(frame, (box[0], box[1] - th - 10), (box[0] + tw, box[1]), box_color, -1)
                                cv2.putText(frame, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            else:
                                prov_color = (160, 160, 160)
                                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), prov_color, 2)
                                label = f"Detecting (track {track_id})"
                                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                cv2.rectangle(frame, (box[0], box[1] - th - 10), (box[0] + tw, box[1]), prov_color, -1)
                                cv2.putText(frame, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    rx1, ry1, rx2, ry2 = self.ROI
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
                    cv2.putText(frame, "TRACK ZONE", (rx1, ry1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                    cv2.polylines(frame, [self.DOOR_POLYGON], True, (0, 0, 255), 2)
                    cx = int(np.mean(self.DOOR_POLYGON[:, 0]))
                    cv2.putText(frame, "DOOR ZONE", (cx, 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    yield frame, current_detections

            except Exception as e:
                import time
                print(f"[{self.cam_id}] stream loi: {e} - thu lai sau 2s")
                time.sleep(2)
                continue
    
    def _face_identify(self, frame, box):
        """Nhận diện mặt trong vùng đầu. Trả (tên, conf, face_box).
        face_box = (fx1, fy1, fx2, fy2) toạ độ mặt trong ẢNH GỐC, hoặc None."""
        if self.face_detector is None or self.face_recognizer is None:
            return None, 0.0, None
        x1, y1, x2, y2 = box
        h_img, w_img = frame.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w_img, int(x2))
        y2 = min(h_img, int(y2))
        h = y2 - y1

        crop = frame[y1:y1 + int(h * 0.6), x1:x2]
        if crop.size == 0:
            return None, 0.0, None
        try:
            faces = self.face_detector.detect(crop)
        except Exception:
            return None, 0.0, None
        print(f"[FACE_DBG] {self.cam_id}: so mat detect duoc = {len(faces)}") 
        if not faces:
            return None, 0.0, None
        face = max(faces, key=lambda f: f["w"] * f["h"])
        print(f"[FACE_DBG] mat to nhat: w={face['w']} h={face['h']}")
        if face["w"] < 25 or face["h"] < 25:
            return None, 0.0, None
        # Toạ độ mặt trong crop → cộng offset (x1, y1) về ảnh gốc
        fx1 = x1 + int(face["x"])
        fy1 = y1 + int(face["y"])
        fx2 = fx1 + int(face["w"])
        fy2 = fy1 + int(face["h"])
        face_img = self.face_detector.align_face(crop, face)
        name, conf = self.face_recognizer.recognize(face_img)
        if name == "Unknown":
            return None, conf, (fx1, fy1, fx2, fy2)
        return name, conf, (fx1, fy1, fx2, fy2)