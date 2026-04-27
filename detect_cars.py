import cv2
import os
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Detect cars in an image using YOLOv8.")
    parser.add_argument("--image", type=str, default=r"C:\Users\jithu\Downloads\20260330112928.jpg", help="Path to the input image.")
    parser.add_argument("--model", type=str, default=r"c:\ai-traffic-control - marl\yolov8\best.pt", help="Path to the YOLOv8 model.")
    parser.add_argument("--conf", type=float, default=0.05, help="Confidence threshold.")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image '{args.image}' not found.")
        return

    print(f"Loading model from {args.model}...")
    model = YOLO(args.model)

    print(f"Running detection on {args.image}...")
    # Read the image
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not read image at '{args.image}'.")
        return
    
    # Perform inference
    results = model(img, conf=args.conf)
    
    # Process results
    for result in results:
        boxes = result.boxes
        
        if boxes is not None and len(boxes) > 0:
            print(f"Detected {len(boxes)} object(s).")
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id]
                print(f"  - {class_name}: {conf:.2f}")
                
            # Plot the results on the image
            annotated_img = result.plot()
            
            # Save the result
            output_filename = f"detected_output.jpg"
            output_path = os.path.join(os.path.dirname(args.image), output_filename)
            cv2.imwrite(output_path, annotated_img)
            print(f"Saved annotated image to {output_path}")
        else:
            print("No objects detected above the confidence threshold.")

if __name__ == "__main__":
    main()
