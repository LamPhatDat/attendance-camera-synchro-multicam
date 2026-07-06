"""
calibrate_zones.py — click the TRACK ZONE (ROI) and DOOR ZONE on a real frame
and get ready-to-paste coordinates for app.py / track.py.

WHY: ROI and door_poly are pixel coordinates of whatever stream you run. They
are camera- and resolution-specific, so they must be drawn on an actual frame
from the same source you'll run live. Calibrating on a scaled screenshot gives
wrong-scale numbers.

USAGE
    # From the live RTSP stream (use the SAME subtype you run in app.py):
    python calibrate_zones.py --source "rtsp://admin:PASS@10.40.90.235:554/cam/realmonitor?channel=1&subtype=1"

    # From a webcam:
    python calibrate_zones.py --source 0

    # From a saved frame (coords only valid if it's native stream resolution):
    python calibrate_zones.py --source frame.png

CONTROLS
    Left click      add a point to the current zone
    d               switch to DOOR mode (polygon, click 3-6 points)
    r               switch to ROI / TRACK ZONE mode (click 2 corners)
    u               undo last point in the current zone
    c               clear the current zone
    p               print current coordinates to the terminal
    s               save an annotated preview image (calibration_preview.png)
    q / Esc         quit and print final coordinates for both zones
"""

import argparse
import sys
import numpy as np
import cv2


def grab_frame(source: str):
    """Return one stable BGR frame from an RTSP url, webcam index, or image path."""
    # Image file?
    img = cv2.imread(source)
    if img is not None:
        return img

    cap = cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: could not open source: {source}")
        sys.exit(1)

    # Read a few frames so the stream/exposure settles, keep the last good one.
    frame = None
    for _ in range(30):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    cap.release()

    if frame is None:
        print("ERROR: could not read a frame from the source.")
        sys.exit(1)
    return frame


class Calibrator:
    def __init__(self, frame, max_w=1280):
        self.frame = frame
        self.h, self.w = frame.shape[:2]
        self.scale = min(1.0, max_w / self.w)
        self.disp_size = (int(self.w * self.scale), int(self.h * self.scale))

        self.mode = "door"  # "door" | "roi"
        self.points = {"door": [], "roi": []}

    # ---- coordinate mapping (display space -> native image space) ----
    def to_image(self, dx, dy):
        return int(round(dx / self.scale)), int(round(dy / self.scale))

    def to_disp(self, x, y):
        return int(round(x * self.scale)), int(round(y * self.scale))

    # ---- mouse ----
    def on_mouse(self, event, dx, dy, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points[self.mode].append(self.to_image(dx, dy))

    # ---- drawing ----
    def render(self):
        canvas = cv2.resize(self.frame, self.disp_size)

        # ROI: bounding rectangle of the first two clicked points
        roi_pts = self.points["roi"]
        if len(roi_pts) >= 2:
            (x1, y1), (x2, y2) = roi_pts[0], roi_pts[1]
            p1 = self.to_disp(min(x1, x2), min(y1, y2))
            p2 = self.to_disp(max(x1, x2), max(y1, y2))
            cv2.rectangle(canvas, p1, p2, (255, 255, 0), 2)
            cv2.putText(canvas, "TRACK ZONE", (p1[0], max(p1[1] - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        for p in roi_pts:
            cv2.circle(canvas, self.to_disp(*p), 4, (255, 255, 0), -1)

        # DOOR: polygon of all clicked points
        door_pts = self.points["door"]
        if len(door_pts) >= 2:
            disp_poly = np.array([self.to_disp(*p) for p in door_pts], dtype=np.int32)
            cv2.polylines(canvas, [disp_poly], len(door_pts) >= 3, (0, 0, 255), 2)
        for p in door_pts:
            cv2.circle(canvas, self.to_disp(*p), 4, (0, 0, 255), -1)

        banner = f"MODE: {self.mode.upper()}   [d]oor  [r]oi  [u]ndo  [c]lear  [p]rint  [s]ave  [q]uit"
        cv2.rectangle(canvas, (0, 0), (self.disp_size[0], 24), (0, 0, 0), -1)
        cv2.putText(canvas, banner, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return canvas

    # ---- output ----
    def roi_tuple(self):
        pts = self.points["roi"]
        if len(pts) < 2:
            return None
        (x1, y1), (x2, y2) = pts[0], pts[1]
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    def print_coords(self):
        print("\n" + "=" * 60)
        print(f"# frame resolution: {self.w} x {self.h}  (coords are in these pixels)")
        roi = self.roi_tuple()
        if roi:
            print(f'"roi": {roi},')
        else:
            print("# ROI: need 2 points (press r, then click two corners)")
        door = self.points["door"]
        if len(door) >= 3:
            pts = ", ".join(f"({x}, {y})" for x, y in door)
            print(f'"door_poly": np.array([{pts}], dtype=np.int32),')
        else:
            print("# DOOR: need >=3 points (press d, then click the door outline)")
        print("=" * 60 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="RTSP url, webcam index (e.g. 0), or image path")
    ap.add_argument("--max-width", type=int, default=1280,
                    help="max on-screen width; does not affect output coords")
    args = ap.parse_args()

    frame = grab_frame(args.source)
    cal = Calibrator(frame, max_w=args.max_width)

    win = "Calibrate zones"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, cal.on_mouse)
    print(__doc__)

    while True:
        cv2.imshow(win, cal.render())
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("d"):
            cal.mode = "door"
        elif key == ord("r"):
            cal.mode = "roi"
        elif key == ord("u") and cal.points[cal.mode]:
            cal.points[cal.mode].pop()
        elif key == ord("c"):
            cal.points[cal.mode].clear()
        elif key == ord("p"):
            cal.print_coords()
        elif key == ord("s"):
            cv2.imwrite("calibration_preview.png", cal.render())
            print("saved calibration_preview.png")

    cv2.destroyAllWindows()
    cal.print_coords()


if __name__ == "__main__":
    main()
