
"""
Face Detection module using YuNet (cv2.FaceDetectorYN).
YuNet is OpenCV's built-in lightweight face detector — no separate download
needed for opencv-contrib-python>=4.8. It runs faster than res10 Caffe SSD,
returns 5-point landmarks (eyes, nose, mouth corners), and works well on
frontal faces typical of kiosk/attendance deployments.
 
Landmarks returned by YuNet per face (indices into the raw row):
  [0..3]   bounding box  x, y, w, h
  [4..5]   right eye     x, y
  [6..7]   left eye      x, y
  [8..9]   nose tip      x, y
  [10..11] right mouth   x, y
  [12..13] left mouth    x, y
  [14]     confidence score
"""
 
import cv2
import numpy as np
import os
import urllib.request
 
 
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
 
 
class FaceDetector:
    """YuNet-based face detector with quality checks."""
 
    def __init__(self, input_size=(320, 320)):
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(model_dir, exist_ok=True)
 
        model_path = os.path.join(model_dir, "face_detection_yunet_2023mar.onnx")
        self._download_if_missing(model_path, YUNET_URL, "YuNet ONNX")
 
        self._input_size = input_size  # (w, h) — updated per frame in detect()
        self.detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=input_size,
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5,
        )
 
        # Quality thresholds
        self.MIN_FACE_RATIO   = 0.15
        self.MAX_FACE_RATIO   = 0.85
        self.BRIGHTNESS_LOW   = 70
        self.BRIGHTNESS_HIGH  = 180
        self.BLUR_THRESHOLD   = 20
        self.CENTER_THRESHOLD = 0.20
 
        print("FaceDetector initialized (YuNet)")
 
    # ── Model download ──────────────────────────────────────────────────────
 
    def _download_if_missing(self, filepath, url, label, timeout=15):
        """Download a model file if it doesn't exist locally."""
        if os.path.exists(filepath):
            return
        print(f"Downloading {label} ...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp, \
                 open(filepath, "wb") as f:
                f.write(resp.read())
            print(f"  [OK] {label} saved to {filepath}")
        except Exception as e:
            print(f"  [FAIL] Could not download {label}: {e}")
            raise RuntimeError(
                f"Required model '{label}' not found and download failed.\n"
                f"Download manually from:\n  {url}\n"
                f"and place it at:\n  {filepath}"
            ) from e
 
    # ── Detection ───────────────────────────────────────────────────────────
 
    def detect(self, frame_bgr):
        """
        Detect faces in a BGR frame.
        Returns list of face dicts:
          {x, y, w, h, confidence, landmarks}
        """
        h, w = frame_bgr.shape[:2]
 
        if (w, h) != self._input_size:
            self._input_size = (w, h)
            self.detector.setInputSize((w, h))
 
        _, raw = self.detector.detect(frame_bgr)
 
        faces = []
        if raw is None:
            return faces
 
        for det in raw:
            x, y, fw, fh = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            x  = max(0, x)
            y  = max(0, y)
            fw = min(fw, w - x)
            fh = min(fh, h - y)
 
            if fw <= 0 or fh <= 0:
                continue
 
            confidence = float(det[14])
            landmarks = {
                "right_eye":   (int(det[4]),  int(det[5])),
                "left_eye":    (int(det[6]),  int(det[7])),
                "nose":        (int(det[8]),  int(det[9])),
                "right_mouth": (int(det[10]), int(det[11])),
                "left_mouth":  (int(det[12]), int(det[13])),
            }
 
            faces.append({
                "x": x, "y": y, "w": fw, "h": fh,
                "confidence": confidence,
                "landmarks": landmarks,
            })
 
        faces.sort(key=lambda f: f["confidence"], reverse=True)
        return faces
 
    # ── Quality checks ──────────────────────────────────────────────────────
 
    def quality_check(self, frame, face):
        """
        Run quality checks on a detected face.
        Returns (passed: bool, reasons: list[str]).
        """
        reasons = []
        h, w = frame.shape[:2]
 
        # --- Face size ---
        face_ratio = face["h"] / h
        if face_ratio < self.MIN_FACE_RATIO:
            reasons.append("Too far")
        if face_ratio > self.MAX_FACE_RATIO:
            reasons.append("Too close")
 
        # --- Centering ---
        cx = (face["x"] + face["w"] / 2) / w
        cy = (face["y"] + face["h"] / 2) / h
        if abs(cx - 0.5) > self.CENTER_THRESHOLD or abs(cy - 0.5) > self.CENTER_THRESHOLD:
            reasons.append("Move to center")
 
        # --- Landmark-based tilt check ---
        lm = face.get("landmarks")
        if lm:
            re, le = lm["right_eye"], lm["left_eye"]
            eye_dx = le[0] - re[0]
            eye_dy = le[1] - re[1]
            tilt_deg = abs(np.degrees(np.arctan2(eye_dy, eye_dx)))
            if tilt_deg > 20:
                reasons.append("Tilt head straight")
 
        # --- Pixel-level checks on the face crop ---
        fx, fy, fw, fh = face["x"], face["y"], face["w"], face["h"]
        face_crop = frame[fy:fy + fh, fx:fx + fw]
 
        if face_crop.size > 0:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
 
            brightness = np.mean(gray)
            if brightness < self.BRIGHTNESS_LOW:
                reasons.append("Too dark")
            elif brightness > self.BRIGHTNESS_HIGH:
                reasons.append("Too bright")
 
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if lap_var < self.BLUR_THRESHOLD:
                reasons.append("Blurry — hold still")
        else:
            reasons.append("Face out of bounds")
 
        return len(reasons) == 0, reasons
 
    # ── Head-pose estimation (dùng cho đăng ký 5 góc) ────────────────────────
 
    def estimate_pose(self, face):
        """Ước lượng góc mặt thô từ 5 landmark. Trả (yaw, pitch).
        yaw<0: quay phải | yaw>0: quay trái | pitch>0: ngước | pitch<0: cúi."""
        lm = face.get("landmarks")
        if not lm:
            return 0.0, 0.0
        re = np.array(lm["right_eye"], dtype=float)
        le = np.array(lm["left_eye"], dtype=float)
        nose = np.array(lm["nose"], dtype=float)
        rm = np.array(lm["right_mouth"], dtype=float)
        lm2 = np.array(lm["left_mouth"], dtype=float)
        eye_center = (re + le) / 2
        mouth_center = (rm + lm2) / 2
        eye_dist = np.linalg.norm(le - re) + 1e-6
        yaw = (nose[0] - eye_center[0]) / eye_dist
        face_v = mouth_center[1] - eye_center[1] + 1e-6
        pitch = 0.5 - (nose[1] - eye_center[1]) / face_v
        return float(yaw), float(pitch)
 
    # ── Crop ────────────────────────────────────────────────────────────────
 
    def crop_face(self, frame, face, padding=0.2):
        """Crop detected face region with padding."""
        h, w = frame.shape[:2]
        fx, fy, fw, fh = face["x"], face["y"], face["w"], face["h"]
 
        pad_w = int(fw * padding)
        pad_h = int(fh * padding)
 
        x1 = max(0, fx - pad_w)
        y1 = max(0, fy - pad_h)
        x2 = min(w, fx + fw + pad_w)
        y2 = min(h, fy + fh + pad_h)
 
        return frame[y1:y2, x1:x2]
 
    # ── Align (tuỳ chọn — căn mặt theo landmark) ─────────────────────────────
 
    def align_face(self, frame, face, target_size=(112, 112)):
        """Align and crop face using 5 landmarks and similarity transform."""
        lm = face.get("landmarks")
        if not lm or len(lm) < 5:
            return self.crop_face(frame, face)
 
        dst_pts = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)
 
        src_pts = np.array([
            lm["right_eye"],
            lm["left_eye"],
            lm["nose"],
            lm["right_mouth"],
            lm["left_mouth"]
        ], dtype=np.float32)
 
        if target_size != (112, 112):
            scale_x = target_size[0] / 112.0
            scale_y = target_size[1] / 112.0
            dst_pts[:, 0] *= scale_x
            dst_pts[:, 1] *= scale_y
 
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        if M is None:
            return self.crop_face(frame, face)   
 
        warped = cv2.warpAffine(frame, M, target_size, flags=cv2.INTER_LINEAR)
        return warped