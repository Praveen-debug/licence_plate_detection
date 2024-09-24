from ultralytics import YOLO
import cv2 as cv
import supervision as sv
import pickle
import numpy as np
import easyocr

class vision():
    def __init__(self, npd_model):
        self.npd_model = YOLO(npd_model)
        self.yolo_model = YOLO("yolov8n.pt")
        self.car_tracker = sv.ByteTrack()
        self.number_plate_tracker = sv.ByteTrack()
        self.map = {}
        self.car_detections = []
        self.number_plate_detection = []
        self.reader = easyocr.Reader(["en"], gpu=False)

    
    # Function to resize bounding_box to a certain resolution
    def resize_bounding_box(self, original_box, original_width, original_height, new_size):
        x, y, w, h = original_box
        new_width, new_height = new_size

        scale_width = new_width / original_width
        scale_height = new_height / original_height

        x_new = int(x * scale_width)
        y_new = int(y * scale_height)
        w_new = int(w * scale_width)
        h_new = int(h * scale_height)

        return (x_new, y_new, w_new, h_new)

    def np_ocr(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        plate = frame[y1:y2, x1:x2]

        plate = cv.cvtColor(plate, cv.COLOR_BGR2GRAY)

        results = self.reader.readtext(plate)
        
        return results


    def save_detections(self, path):
        with open(path, "wb") as f:
            car_detections_cpu = [det.cpu() for det in self.car_detections]
            number_plate_detections_cpu = [det.cpu() for det in self.number_plate_detection]
            pickle.dump({
                "cars": car_detections_cpu,
                "number_plates": number_plate_detections_cpu,
            }, f)

    def load_detections(self, path):
        with open(path, "rb") as f:
            detections = pickle.load(f)  
        return detections
    
    # Function to compute intersection over union
    def compute_iou(self, bbox_car, bbox_np):
        x1 = max(bbox_car[0], bbox_np[0])
        y1 = max(bbox_car[1], bbox_np[1])
        x2 = min(bbox_car[2], bbox_np[2])
        y2 = min(bbox_car[3], bbox_np[3])

        area_intersection = max(0, (x2 - x1)) * max(0, (y2- y1))

        car_area= (bbox_car[2] - bbox_car[0]) * (bbox_car[3] - bbox_car[1])
        np_area = (bbox_np[2] - bbox_np[0]) * (bbox_np[3] - bbox_np[1])

        union = float(car_area + np_area - area_intersection)

        return area_intersection / union



    def detect_number_plate(self, img):
        results = self.npd_model.predict(img)[0]
        self.number_plate_detection.append(results)
        return results
    
    def detect_cars(self, img):
        results = self.yolo_model.predict(img)[0]
        self.car_detections.append(results)
        return results

    def track_objects(self, results, is_car):
        detections = sv.Detections.from_ultralytics(results)

        if is_car:
            detections_with_track_id = self.car_tracker.update_with_detections(detections)
        else:
            detections_with_track_id = self.number_plate_tracker.update_with_detections(detections)

        return detections_with_track_id

    

    def draw_bounding_car_with_number_plate(self, detections, number_plates, frame):
        for car in detections:
            car_bbox = car[0].tolist()
            x1, y1, x2, y2 = map(int, car_bbox)

            for plate in number_plates:
                plate_bbox = plate[0].tolist()
                n1, m1, n2, m2 = map(int, plate_bbox)

                if n1 >= x1 and m1 >= y1 and n2 <= x2 and m2 <= y2:

                    car_track_id = car[4]
                    plate_track_id = plate[4]

                    self.map[str(car_track_id)] = str(plate_track_id)

                    center = int((x2 + x1) / 2)

                    coordinates = np.array([[center - 10, y1-20], [center + 10, y1-20], [center, y1-10]])
                    cv.polylines(frame, [coordinates], isClosed=True, color=(0, 0, 0), thickness=2)
                    cv.fillPoly(frame, [coordinates],color=(0, 255, 0))
                    cv.rectangle(frame, (n1, m1), (n2, m2), (0, 255, 0), 2)

                    break

        return frame


# Realtime 👇


if __name__ == "__main__":
    cap = cv.VideoCapture("./Videos/traffic 1.mp4")
    Detector = vision("./Weights/Best Weight.pt")

    fps = 30
    wait_time = int(1000 / fps)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break


        frame = cv.resize(frame, (640, 480))
        
        results = Detector.detect_cars(frame)

        cars = Detector.track_objects(results, True)

        results = Detector.detect_number_plate(frame)

        number_plates = Detector.track_objects(results, False)

        frame = Detector.draw_bounding_car_with_number_plate(cars, number_plates, frame)

        cv.imshow("Frame", frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

cv.destroyAllWindows()



# Loading from detections 👇

# if __name__ == "__main__":
#     cap = cv.VideoCapture("./Videos/traffic 1.mp4")
#     Detector = vision("./Weights/Online Downloaded.pt")

#     detections = Detector.load_detections("./detections/traffic_1_detections_main_2.pkl")
#     cars = detections["cars"]
#     number_plates = detections["number_plates"]

#     fps = 30

#     wait_time = int(1000 / fps)

#     while True:
#         ret, frame = cap.read()

#         og_frame = frame.copy()

#         frame = cv.resize(frame, (640, 480))
#         current_frame_number = int(cap.get(cv.CAP_PROP_POS_FRAMES))

#         car = Detector.track_objects(cars[current_frame_number], True)
#         plate = Detector.track_objects(number_plates[current_frame_number], False) 

        
#         frame = frame = Detector.draw_bounding_car_with_number_plate(car, plate, frame)

#         cv.imshow("Window", frame)

#         if cv.waitKey(wait_time) & 0xFF == ord("q"):
#             break