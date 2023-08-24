import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

import supervision as sv


import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from email_settings import password, from_email, to_email

    
    
# create server
server = smtplib.SMTP('smtp.gmail.com: 587')

server.starttls()

# Login Credentials for sending the mail
server.login(from_email, password)

    
def send_email(to_email, from_email, people_detected=1):
 
    message = MIMEMultipart()
    message['From'] = from_email
    message['To'] = to_email
    message['Subject'] = "Security Alert"
    # add in the message body
    message.attach(MIMEText(f'ALERT - {people_detected} persons has been detected!!', 'plain'))
    server.sendmail(from_email, to_email, message.as_string())

    


class ObjectDetection:

    def __init__(self, capture_index):
       
        self.capture_index = capture_index
        
        self.email_sent = False
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("Using Device: ", self.device)
        
        self.model = self.load_model()
        
        self.CLASS_NAMES_DICT = self.model.model.names
    
        self.box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.default(), thickness=3, text_thickness=3, text_scale=1.5)
    

    def load_model(self):
       
        model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n model
        model.fuse()
    
        return model


    def predict(self, frame):
       
        results = self.model(frame)
        
        return results
    

    def plot_bboxes(self, results, frame):
        
        xyxys = []
        confidences = []
        class_ids = []
        
        # Extract detections for person class
        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)
            
            if class_id == 0:
                
                xyxys.append(result.boxes.xyxy.cpu().numpy())
                confidences.append(result.boxes.conf.cpu().numpy())
                class_ids.append(result.boxes.cls.cpu().numpy().astype(int))

        
            
        # Setup detections for visualization
        
        detections = sv.Detections.from_ultralytics(results[0])
    
        frame = self.box_annotator.annotate(scene=frame, detections=detections)
        
        
        return frame, class_ids
    
    
    
    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
      
        while True:
          
            start_time = time()
            
            ret, frame = cap.read()
            assert ret
            
            results = self.predict(frame)
            frame, class_ids = self.plot_bboxes(results, frame)
            
            if len(class_ids) > 0:
                if not self.email_sent:  # Only send email if it hasn't been sent for the current detection
                    send_email(to_email, from_email, len(class_ids))
                    self.email_sent = True  # Set the flag to True after sending the email
            else:
                self.email_sent = False  # Reset the flag when no person is detected

            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
             
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
            cv2.imshow('YOLOv8 Detection', frame)
            
            frame_count += 1
 
            if cv2.waitKey(5) & 0xFF == 27:
                
                break
        
        cap.release()
        cv2.destroyAllWindows()
        server.quit()
        
        
    
detector = ObjectDetection(capture_index=0)
detector()