"""
Webcam-based world-frame object tracking.

- Detects non-human objects with YOLO segmentation (COCO classes).
- Masks humans before computing inter-frame homography (ORB + RANSAC).
- Maintains a cumulative homography to frame 0 -> "world" coordinates.
- Associates detections across frames in WORLD space so IDs persist even
  when objects leave and re-enter the view.
- Draws contours on detected non-human objects and overlays their world
  coordinates near each detection.

Requires: pip install ultralytics opencv-python numpy
"""

import cv2
import numpy as np
from ultralytics import YOLO

# ------------------------- Config -------------------------
# MODEL_NAME       = "yolo11n-seg.pt"   # auto-downloads on first run
MODEL_NAME       = "yolo26m-seg.pt"
CONF_THRESH      = 0.4
PERSON_CLASS_ID  = 0                  # COCO 'person'
ASSOC_DIST       = 60.0               # px in world space for re-association
EMA_ALPHA        = 0.3                # smoothing for world position updates
MIN_MATCHES      = 25                 # min ORB matches for a valid homography
RANSAC_REPROJ    = 3.0
ANCHOR_EVERY     = 30                 # try re-anchoring to frame 0 every N frames
ORB_FEATURES     = 2000
# ----------------------------------------------------------


class WorldTrack:
    """A single persistent object track living in world (frame-0) coordinates."""
    _next_id = 0

    def __init__(self, world_xy, cls_id, cls_name):
        self.id = WorldTrack._next_id
        WorldTrack._next_id += 1
        self.world_xy = np.asarray(world_xy, dtype=np.float32)
        self.cls_id = cls_id
        self.cls_name = cls_name
        self.hits = 1
        self.last_seen_frame = 0

    def update(self, world_xy, frame_idx):
        self.world_xy = (1 - EMA_ALPHA) * self.world_xy + EMA_ALPHA * np.asarray(
            world_xy, dtype=np.float32
        )
        self.hits += 1
        self.last_seen_frame = frame_idx


def apply_H(H, pts_xy):
    """Apply a 3x3 homography to Nx2 points; returns Nx2."""
    pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    homo = np.hstack([pts, ones])                 # Nx3
    out = (H @ homo.T).T                          # Nx3
    out = out[:, :2] / out[:, 2:3]
    return out


def estimate_homography(gray_a, gray_b, mask_a, mask_b, orb, matcher):
    """
    Estimate H mapping points in gray_b -> gray_a.
    Masks are 0/255 uint8; features are extracted only where mask != 0.
    (So pass background-only masks, i.e. NOT person.)
    """
    kp_a, des_a = orb.detectAndCompute(gray_a, mask_a)
    kp_b, des_b = orb.detectAndCompute(gray_b, mask_b)
    if des_a is None or des_b is None or len(kp_a) < MIN_MATCHES or len(kp_b) < MIN_MATCHES:
        return None

    matches = matcher.match(des_a, des_b)
    if len(matches) < MIN_MATCHES:
        return None
    matches = sorted(matches, key=lambda m: m.distance)[: max(MIN_MATCHES * 4, 200)]

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches])

    H, inliers = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, RANSAC_REPROJ)
    if H is None or inliers is None or inliers.sum() < MIN_MATCHES:
        return None
    return H


def build_person_mask(result, shape_hw):
    """Union all person instance masks, resized to the frame shape. Returns 0/255 uint8."""
    h, w = shape_hw
    person_mask = np.zeros((h, w), dtype=np.uint8)
    if result.masks is None or result.boxes is None:
        return person_mask
    cls_arr = result.boxes.cls.cpu().numpy().astype(int)
    mdata = result.masks.data.cpu().numpy()   # (N, Hm, Wm) in [0,1]
    for i, c in enumerate(cls_arr):
        if c != PERSON_CLASS_ID:
            continue
        m = mdata[i]
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        person_mask[m > 0.5] = 255
    # Dilate so feature points on the person's silhouette are also excluded.
    person_mask = cv2.dilate(person_mask, np.ones((15, 15), np.uint8))
    return person_mask


def associate(tracks, detections, frame_idx):
    """
    Greedy nearest-neighbour association in world space, class-gated.
    detections: list of dicts with keys 'world_xy', 'cls_id', 'cls_name'.
    Returns list of (track, detection) pairs and list of unmatched detections.
    """
    assigned_tracks = set()
    assigned_dets = set()
    pairs = []

    # Build a cost table (Euclidean distance, gated by class & threshold).
    candidates = []
    for di, det in enumerate(detections):
        for ti, tr in enumerate(tracks):
            if tr.cls_id != det["cls_id"]:
                continue
            d = float(np.linalg.norm(tr.world_xy - det["world_xy"]))
            if d <= ASSOC_DIST:
                candidates.append((d, di, ti))

    candidates.sort(key=lambda x: x[0])
    for d, di, ti in candidates:
        if di in assigned_dets or ti in assigned_tracks:
            continue
        assigned_dets.add(di)
        assigned_tracks.add(ti)
        pairs.append((tracks[ti], detections[di]))

    unmatched = [det for di, det in enumerate(detections) if di not in assigned_dets]
    for tr, det in pairs:
        tr.update(det["world_xy"], frame_idx)
    for det in unmatched:
        tracks.append(WorldTrack(det["world_xy"], det["cls_id"], det["cls_name"]))

    return tracks


def main():
    model = YOLO(MODEL_NAME)
    names = model.names

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    orb = cv2.ORB_create(ORB_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    tracks = []
    H_cumulative = np.eye(3, dtype=np.float32)   # current frame -> frame 0
    prev_gray = None
    prev_bg_mask = None
    frame0_gray = None
    frame0_bg_mask = None

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1) Run YOLO segmentation on the current frame.
        result = model.predict(frame, conf=CONF_THRESH, verbose=False)[0]

        # 2) Build person mask (for excluding from feature matching).
        person_mask = build_person_mask(result, (h, w))
        bg_mask = cv2.bitwise_not(person_mask)

        # 3) Homography update.
        if frame_idx == 0:
            frame0_gray = gray.copy()
            frame0_bg_mask = bg_mask.copy()
            H_cumulative = np.eye(3, dtype=np.float32)
        else:
            H_rel = estimate_homography(prev_gray, gray, prev_bg_mask, bg_mask,
                                        orb, matcher)
            if H_rel is not None:
                H_cumulative = H_cumulative @ H_rel
            # Periodic re-anchoring to frame 0 limits drift.
            if frame_idx % ANCHOR_EVERY == 0:
                H_anchor = estimate_homography(frame0_gray, gray,
                                               frame0_bg_mask, bg_mask,
                                               orb, matcher)
                if H_anchor is not None:
                    H_cumulative = H_anchor

        # 4) Collect non-human detections + draw contours.
        detections = []
        vis = frame.copy()
        if result.masks is not None and result.boxes is not None:
            cls_arr = result.boxes.cls.cpu().numpy().astype(int)
            conf_arr = result.boxes.conf.cpu().numpy()
            polygons = result.masks.xy   # list of Nx2 arrays (image coords)

            for i, cls_id in enumerate(cls_arr):
                # if cls_id == PERSON_CLASS_ID: # exclude human object
                #     continue
                poly = polygons[i]
                if poly is None or len(poly) < 3:
                    continue
                contour = poly.astype(np.int32).reshape(-1, 1, 2)
                cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)

                # Use the mask centroid as the detection anchor (robust to bbox bg).
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    cx, cy = poly.mean(axis=0)
                else:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                img_xy = np.array([cx, cy], dtype=np.float32)

                world_xy = apply_H(H_cumulative, img_xy)[0]
                detections.append({
                    "cls_id": int(cls_id),
                    "cls_name": names[int(cls_id)],
                    "conf": float(conf_arr[i]),
                    "img_xy": img_xy,
                    "world_xy": world_xy,
                })

        # 5) Associate to world-frame tracks.
        tracks = associate(tracks, detections, frame_idx)

        # 6) Overlay: for each detection, label with ID and world coordinates.
        for det in detections:
            # Find the matching track (by nearest class + position).
            matched = None
            for tr in tracks:
                if tr.cls_id == det["cls_id"] and \
                   np.linalg.norm(tr.world_xy - det["world_xy"]) <= ASSOC_DIST:
                    matched = tr
                    break
            cx, cy = det["img_xy"].astype(int)
            wx, wy = det["world_xy"]
            tid = matched.id if matched is not None else -1
            label = f"{det['cls_name']} #{tid}"
            coord = f"W=(X:{wx:+.0f}, Y:{wy:+.0f})"
            cv2.circle(vis, (cx, cy), 4, (0, 255, 255), -1)
            cv2.putText(vis, label, (cx + 6, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(vis, coord, (cx + 6, cy + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # 7) HUD.
        cv2.putText(vis, f"frame={frame_idx}  tracks={len(tracks)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # Shade persons so you can see what was excluded from feature matching.
        if person_mask.any():
            overlay = vis.copy()
            overlay[person_mask > 0] = (0, 0, 255)
            vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)

        cv2.imshow("World-frame tracking (q to quit)", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        prev_gray = gray
        prev_bg_mask = bg_mask
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()