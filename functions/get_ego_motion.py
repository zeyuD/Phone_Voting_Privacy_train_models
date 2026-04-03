import cv2
import numpy as np

def track_camera_motion(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Intrinsic parameters (Estimates if unknown)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    focal = width  # Approximation
    pp = (width / 2, height / 2)

    # Initialize ORB detector and matcher
    orb = cv2.ORB_create(3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)

    # Global Pose (Position and Orientation)
    cur_R = np.eye(3)
    cur_t = np.zeros((3, 1))
    trajectory = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = orb.detectAndCompute(gray, None)
        
        if des2 is not None:
            # Match features between frames
            matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

            # Calculate movement (Essential Matrix)
            E, _ = cv2.findEssentialMat(pts2, pts1, focal, pp, cv2.RANSAC, 0.999, 1.0)
            _, R, t, _ = cv2.recoverPose(E, pts2, pts1, focal=focal, pp=pp)

            # Accumulate movement into global coordinates
            cur_t = cur_t + cur_R.dot(t)
            cur_R = R.dot(cur_R)
            
            # Save time series data
            trajectory.append({'rotation': cur_R, 'position': cur_t})

        kp1, des1 = kp2, des2

    cap.release()
    return trajectory