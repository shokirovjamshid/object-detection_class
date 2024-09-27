import torchvision.transforms as transforms
import cv2
from  fasterrcnn_model import model,preprocess,weights
import torch
#  Video Capture
cap = cv2.VideoCapture('cars.mp4')
while True:
    ret, frame = cap.read()
    if ret == True:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        transform = transforms.ToTensor()
        frame_tensor = transform(frame_rgb)
        batch = [preprocess(frame_tensor)]
        # Step 4: Use the model and visualize the prediction
        with torch.no_grad():  # Model baholashda backpropagation qilmasligi uchun
            prediction = model(batch)[0]
        categories = weights.meta["categories"]
        boxes = prediction['boxes']
        labels = prediction["labels"]
        for i in range(len(labels)):
            x1 = int(boxes[i][0].item()) # Convert x1 to integer
            y1 = int(boxes[i][1].item()) # Convert y1 to integer
            x2 = int(boxes[i][2].item()) # Convert x2 to integer
            y2 = int(boxes[i][3].item())
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,categories[labels[i]],(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow('image',frame)
        if cv2.waitKey(1)  == ord('q'):
            break
    else:
        break
    

cap.release()
cv2.destroyAllWindows()