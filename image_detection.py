import cv2, os
from imageai.Detection import ObjectDetection
from imageai.Detection.Custom import CustomVideoObjectDetection
from imageai.Detection.Custom import CustomObjectDetection


class Detect: 

    def __init__(self, custom_model=False):

        self.execution_path = os.getcwd()
        
        

        if custom_model == True: 
            model = r"C:\Users\Matt\OneDrive\GitHub\PersonDetection\pics\models\tiny-yolov3_pics_last.pt"
            json = r"C:\Users\Matt\OneDrive\GitHub\PersonDetection\pics\json\pics_tiny-yolov3_detection_config.json"
            # json = r"C:\Users\Matt\OneDrive\GitHub\SmartMirror\SmartMirror\pics\json\pics_tiny-yolov3_detection_config.json"
            # model = r"C:\Users\Matt\OneDrive\GitHub\SmartMirror\SmartMirror\pics\models\tiny-yolov3_pics_mAP-0.01951_epoch-107.pt"
            self.detector = CustomVideoObjectDetection()
            self.detector.setModelTypeAsTinyYOLOv3()
            self.detector.setJsonPath(json)
            self.detector.setModelPath(model)
            
        else: 
            model = r"C:\Users\Matt\OneDrive\GitHub\SmartMirror\SmartMirror\pics\models\tiny-yolov3.pt"
            self.detector.setModelTypeAsTinyYOLOv3()
            self.detector.setModelPath(os.path.join(self.execution_path , "./models/tiny-yolov3.pt"))

        self.detector.loadModel()
        self.camera = cv2.VideoCapture(0)

    
    def new_label(self): 
        self.detector.detectObjectsFromVideo(   
                frames_per_second=20, 
                log_progress=True,
                camera_input = self.camera,
                save_detected_video=True,
                output_file_path='test',
                detection_timeout=20,
            )
    
    def label(self): 
        while True:
            ret, img = self.camera.read()
            self.detector.detectObjectsFromvideo(camera_input=self.camera, output_file_path=os.path.join(self.execution_path , "camera_detected_video"), frames_per_second=20, log_progress=True)
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

        # setup minio client
        # client = Minio(
        #     "localhost:9000",
        #     access_key="minioadmin",
        #     secret_key="minioadmin",
        #     secure=False,
        # )