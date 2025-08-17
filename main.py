import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
from collections import Counter

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO live object counting")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    # Webcam setup
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # change to 1 if external cam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Load YOLOv8n model
    model = YOLO("yolov8n.pt")  # replace with your trained weights if needed

    # BoxAnnotator in 0.26.1 has no labels argument
    box_annotator = sv.BoxAnnotator()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Draw bounding boxes (no labels in this function)
        frame = box_annotator.annotate(scene=frame, detections=detections)

        # Draw class names + confidence manually
        for i, (class_id, conf, xyxy) in enumerate(
            zip(detections.class_id, detections.confidence, detections.xyxy)
        ):
            name = model.model.names[int(class_id)]
            label = f"{name} {conf:.2f}"
            x1, y1, _, _ = map(int, xyxy)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Count per-class
        class_counts = Counter(model.model.names[int(cls)] for cls in detections.class_id)

        # Show counts on the side
        y_offset = 40
        for cls_name, count in class_counts.items():
            cv2.putText(frame, f"{cls_name}: {count}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            y_offset += 40

        # Total count
        total_count = sum(class_counts.values())
        cv2.putText(frame, f"Total Objects: {total_count}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        # Display output
        cv2.imshow("YOLOv8n Object Counter", frame)

        if cv2.waitKey(30) == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()