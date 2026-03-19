import cv2
import imutils
import numpy as np


def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


def detect_plate(frame):
    """
    Returns:
        plate_contour: 4-point contour or None
        debug_frame: frame with drawn candidate
    """
    debug_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    edged = cv2.Canny(gray, 30, 200)

    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    best_quad = None
    best_area = 0

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            aspect_ratio = w / float(h) if h > 0 else 0

            # rough plate-like shape
            if 2.0 <= aspect_ratio <= 6.5 and area > 2000:
                if area > best_area:
                    best_area = area
                    best_quad = approx

    if best_quad is not None:
        cv2.drawContours(debug_frame, [best_quad], -1, (0, 255, 0), 2)
        pts = best_quad.reshape(4, 2)
        return order_points(pts), debug_frame

    return None, debug_frame