import cv2, os
from imageai.Detection import ObjectDetection
import os
execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path , "yolov3.pt"))
detector.loadModel()
camera = cv2.VideoCapture(0)
# camera.set(cv2.CAP_PROP_FRAME_WIDTH, 650)
# camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 750)

while True:
    ret, img = camera.read()
    annot_image, preds = detector.detectObjectsFromVideo( 
        input_image=img,
        # input_type="array",
        output_type="array",
        display_percentage_probability=True,
        display_object_name=True,
        
    )
    cv2.imshow("annotated", annot_image)

    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
        break

camera.release()
cv2.destroyAllWindows()

