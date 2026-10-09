import cv2
import numpy as np
from ultralytics import YOLO

class SmartCropper:
    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.35):
        # Load lightweight nano model for person detection (class 0)
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold

    def calculate_crop_centers(self, video_path: str, start_time: float, duration: float, sample_fps: int = 5, alpha: float = 0.15):
        """
        Samples frames across the segment window, detects primary person bounding box with confidence threshold,
        applies Exponential Moving Average (EMA) smoothing and center fallbacks.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Target 9:16 width given source height
        target_crop_width = int(height * (9 / 16))
        max_x = max(0, width - target_crop_width)
        default_crop_x = max_x // 2

        start_frame = int(start_time * fps)
        total_frames = int(duration * fps)
        step = max(1, int(fps / sample_fps))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        ema_crop_x = float(default_crop_x)
        crop_centers = []

        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i)
            ret, frame = cap.read()
            if not ret:
                break

            # Run person detection
            results = self.model(frame, classes=[0], verbose=False)
            boxes = results[0].boxes

            # Filter high confidence boxes
            valid_boxes = [b for b in boxes if float(b.conf[0]) >= self.conf_threshold] if len(boxes) > 0 else []

            if valid_boxes:
                # Pick largest detected person box
                largest_box = max(valid_boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
                x1, _, x2, _ = largest_box.xyxy[0].tolist()
                box_center_x = (x1 + x2) / 2.0
                target_x = box_center_x - (target_crop_width / 2.0)
            else:
                target_x = float(default_crop_x)

            # Clamp target crop_x strictly within [0, max_x]
            clamped_target_x = max(0.0, min(target_x, float(max_x)))

            # Exponential Moving Average (EMA) smoothing
            ema_crop_x = alpha * clamped_target_x + (1 - alpha) * ema_crop_x
            crop_centers.append(ema_crop_x)

        cap.release()

        if not crop_centers:
            final_crop_x = default_crop_x
        else:
            final_crop_x = int(round(np.median(crop_centers)))

        # Final strict clamp
        final_crop_x = max(0, min(final_crop_x, max_x))
        return final_crop_x, target_crop_width, height
