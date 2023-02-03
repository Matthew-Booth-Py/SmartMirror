import cv2, os
from imageai.Detection import ObjectDetection


class Detect: 

    def __init__(self):
        self.execution_path = os.getcwd()
        self.detector = ObjectDetection()
        self.detector.setModelTypeAsYOLOv3()
        self.detector.setModelPath(os.path.join(self.execution_path , "models/yolov3.pt"))
        self.detector.loadModel()
        self.camera = self.create_image()

        
    def create_image(self): 
        camera = cv2.VideoCapture(0)
        return camera
    
    def label(self): 
        while True:
            ret, img = self.camera.read()
            annot_image, preds = self.detector.detectObjectsFromImage( 
                input_image=img,
                # input_type="array",
                output_type="array",
                display_percentage_probability=True,
                display_object_name=True,
                
            )
            cv2.imshow("annotated", annot_image)

            if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
                break

        self.camera.release()
        cv2.destroyAllWindows()