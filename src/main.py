import os
import cv2

from camera import open_camera
from detect import detect_plate
from align import align_plate
from ocr import read_plate_text
from validate import is_valid_plate
from temporal import TemporalConfirm
from storage import PlateStorage


def save_debug_screenshots(frame, aligned, ocr_img):
    os.makedirs("screenshots", exist_ok=True)

    cv2.imwrite("screenshots/detection.png", frame)

    if aligned is not None:
        cv2.imwrite("screenshots/alignment.png", aligned)

    if ocr_img is not None:
        cv2.imwrite("screenshots/ocr.png", ocr_img)


def main():
    cap = open_camera()

    temporal = TemporalConfirm(max_history=10, confirm_threshold=3)
    storage = PlateStorage("data/plates.csv")

    print("Press 's' to save screenshots")
    print("Press 'q' to quit")

    last_aligned = None
    last_ocr_img = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error")
            break

        # ===== 1. DETECTION =====
        plate_points, debug_frame = detect_plate(frame)

        aligned_plate = None
        ocr_img = None
        plate_text = ""
        valid = False
        confirmed = None

        # ===== 2. ALIGNMENT =====
        if plate_points is not None:
            aligned_plate = align_plate(frame, plate_points)

            # ===== 3. OCR =====
            if aligned_plate is not None:
                plate_text, ocr_img = read_plate_text(aligned_plate)

                # ===== 4. VALIDATION =====
                valid = is_valid_plate(plate_text)

                # ===== 5. TEMPORAL CONFIRM =====
                if valid:
                    confirmed = temporal.update(plate_text)

                    # ===== 6. SAVE =====
                    if confirmed:
                        saved = storage.save_plate(confirmed)
                        if saved:
                            print(f"[SAVED] {confirmed}")

        # ===== DISPLAY =====
        display_frame = debug_frame.copy()

        status = f"OCR: {plate_text if plate_text else 'N/A'} | VALID: {valid}"
        cv2.putText(display_frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if confirmed:
            cv2.putText(display_frame, f"CONFIRMED: {confirmed}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

        cv2.imshow("Detection", display_frame)

        if aligned_plate is not None:
            cv2.imshow("Aligned Plate", aligned_plate)
            last_aligned = aligned_plate

        if ocr_img is not None:
            cv2.imshow("OCR Input", ocr_img)
            last_ocr_img = ocr_img

        key = cv2.waitKey(1) & 0xFF

        # save screenshots
        if key == ord('s'):
            save_debug_screenshots(display_frame, last_aligned, last_ocr_img)
            print("[INFO] Screenshots saved")

        # quit
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()