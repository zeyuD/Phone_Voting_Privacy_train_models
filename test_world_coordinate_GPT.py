import cv2
import numpy as np
from ultralytics import YOLO


# ----------------------------
# Configuration
# ----------------------------
CAMERA_INDEX = 0
# MODEL_NAME = "yolov8n.pt"   # auto-downloads if needed
MODEL_NAME = "yolo26m.pt"
CONF_THRES = 0.35
IOU_THRES = 0.45

# Tracking / pseudo-world settings
MATCH_DIST_THRES = 80.0      # pixels in pseudo-world frame
MAX_MISSED_FRAMES = 45       # keep stale track for this many frames
MIN_POINTS_FOR_MOTION = 12

# Drawing
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 255)
CENTER_COLOR = (0, 0, 255)
HUMAN_CLASS_NAME = "person"


# ----------------------------
# Helpers
# ----------------------------
def affine_2x3_to_3x3(A):
    """Convert OpenCV 2x3 affine matrix to 3x3 homogeneous."""
    H = np.eye(3, dtype=np.float32)
    H[:2, :] = A
    return H


def transform_point(H, pt):
    """Apply 3x3 homogeneous transform to a 2D point."""
    x, y = pt
    p = np.array([x, y, 1.0], dtype=np.float32)
    q = H @ p
    if abs(q[2]) < 1e-6:
        return float(q[0]), float(q[1])
    return float(q[0] / q[2]), float(q[1] / q[2])


def create_person_mask(frame_shape, detections, names):
    """
    Build a mask where background = 255 (usable for motion estimation),
    person boxes = 0 (excluded from camera-motion estimation).
    """
    h, w = frame_shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)

    if detections.boxes is None:
        return mask

    for box in detections.boxes:
        cls_id = int(box.cls[0].item())
        cls_name = names.get(cls_id, str(cls_id))
        if cls_name == HUMAN_CLASS_NAME:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

    return mask


def detect_nonhuman_objects(model, frame, conf=CONF_THRES, iou=IOU_THRES):
    """
    Returns the first YOLO result and a list of non-human detections:
    [{
        'bbox': (x1, y1, x2, y2),
        'center': (cx, cy),
        'class_id': int,
        'class_name': str,
        'conf': float
    }, ...]
    """
    results = model.predict(source=frame, conf=conf, iou=iou, verbose=False)
    result = results[0]
    names = result.names

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = names.get(cls_id, str(cls_id))
            score = float(box.conf[0].item())

            if cls_name == HUMAN_CLASS_NAME:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "class_id": cls_id,
                "class_name": cls_name,
                "conf": score,
            })

    return result, detections


def estimate_camera_motion(prev_gray, curr_gray, bg_mask):
    """
    Estimate motion from current frame -> previous frame using background features.
    Returns 3x3 H_step such that:
        p_prev ~= H_step @ p_curr
    """
    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=300,
        qualityLevel=0.01,
        minDistance=8,
        mask=bg_mask
    )

    if prev_pts is None or len(prev_pts) < MIN_POINTS_FOR_MOTION:
        return np.eye(3, dtype=np.float32)

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    if curr_pts is None or status is None:
        return np.eye(3, dtype=np.float32)

    good_prev = prev_pts[status.flatten() == 1].reshape(-1, 2)
    good_curr = curr_pts[status.flatten() == 1].reshape(-1, 2)

    if len(good_prev) < MIN_POINTS_FOR_MOTION:
        return np.eye(3, dtype=np.float32)

    # Estimate transform from current -> previous
    A, inliers = cv2.estimateAffinePartial2D(
        good_curr,
        good_prev,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
        refineIters=10
    )

    if A is None:
        return np.eye(3, dtype=np.float32)

    return affine_2x3_to_3x3(A)


# ----------------------------
# Simple track manager
# ----------------------------
class TrackManager:
    def __init__(self, match_dist=MATCH_DIST_THRES, max_missed=MAX_MISSED_FRAMES):
        self.next_id = 1
        self.tracks = {}  # track_id -> dict
        self.match_dist = match_dist
        self.max_missed = max_missed

    def update(self, detections, H_ref_from_curr, frame_idx):
        """
        For each detection, compute pseudo-world coord in reference frame,
        then associate by nearest neighbor in that frame.
        """
        assigned_track_ids = set()

        for det in detections:
            cx, cy = det["center"]
            wx, wy = transform_point(H_ref_from_curr, (cx, cy))
            det["world_xy"] = (wx, wy)

            best_id = None
            best_dist = float("inf")

            for tid, tr in self.tracks.items():
                if tr["class_name"] != det["class_name"]:
                    continue
                if tid in assigned_track_ids:
                    continue

                tx, ty = tr["world_xy"]
                dist = np.hypot(wx - tx, wy - ty)
                if dist < best_dist and dist < self.match_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1

            self.tracks[best_id] = {
                "world_xy": (wx, wy),
                "bbox": det["bbox"],
                "class_name": det["class_name"],
                "conf": det["conf"],
                "last_seen": frame_idx,
            }
            det["track_id"] = best_id
            assigned_track_ids.add(best_id)

        # Remove very stale tracks
        to_delete = []
        for tid, tr in self.tracks.items():
            if frame_idx - tr["last_seen"] > self.max_missed:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        return detections


# ----------------------------
# Main
# ----------------------------
def main():
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame from webcam.")

    ref_frame = frame0.copy()
    prev_frame = frame0.copy()
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # H_ref_from_curr maps current-frame points into the reference-frame coordinates
    H_ref_from_prev = np.eye(3, dtype=np.float32)

    tracker = TrackManager()
    frame_idx = 0

    print("Press 'q' to quit.")
    print("Press 'r' to reset reference frame.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect objects in current frame
        yolo_result, nonhuman_dets = detect_nonhuman_objects(model, frame)

        # Mask out humans when estimating background motion
        person_mask = create_person_mask(frame.shape, yolo_result, yolo_result.names)

        # Estimate motion current -> previous from background
        H_prev_from_curr = estimate_camera_motion(prev_gray, curr_gray, person_mask)

        # Accumulate current -> reference
        H_ref_from_curr = H_ref_from_prev @ H_prev_from_curr

        # Update tracks in reference frame
        tracked = tracker.update(nonhuman_dets, H_ref_from_curr, frame_idx)

        # Draw current detections and pseudo-world coords
        for det in tracked:
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = det["center"]
            wx, wy = det["world_xy"]
            tid = det["track_id"]
            cls_name = det["class_name"]
            conf = det["conf"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
            cv2.circle(frame, (int(cx), int(cy)), 4, CENTER_COLOR, -1)

            line1 = f"{cls_name}  ID:{tid}  {conf:.2f}"
            line2 = f"W=({wx:.1f}, {wy:.1f})"

            y_text = max(20, y1 - 10)
            cv2.putText(frame, line1, (x1, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, BOX_COLOR, 2, cv2.LINE_AA)
            cv2.putText(frame, line2, (x1, y_text + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 2, cv2.LINE_AA)

        # Small HUD
        cv2.putText(frame, "Pseudo-world frame = reference image", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "q: quit   r: reset reference", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Static Non-Human Object Coordinates", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            # Reset pseudo-world frame to current frame
            ref_frame = frame.copy()
            ref_gray = curr_gray.copy()
            H_ref_from_prev = np.eye(3, dtype=np.float32)
            tracker = TrackManager()
            print("Reference frame reset.")

        # Advance
        prev_gray = curr_gray
        prev_frame = frame.copy()
        H_ref_from_prev = H_ref_from_curr

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()