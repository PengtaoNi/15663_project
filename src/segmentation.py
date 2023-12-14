import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
from torchvision import transforms
from mtcnn import MTCNN

def face_bound(image, debug=False):
    H, W = image.shape[:2]

    detector = MTCNN()
    faces = detector.detect_faces(image)

    out = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for face in faces:
        x, y, w, h = face['box']

        # extend the bounds to cover hair and chin
        x = max(x-w//2, 0)
        w = min(w*2, W-x)
        y = max(y-h, 0)
        h = min(int(h*2.1), H-y)

        out[y:y+h, x:x+w] = 1

    if debug:
        image = np.where(out[:,:,None] == 1, image, image//8)
        plt.imshow(image)
        plt.show()

    return out

def subject_mask(image, debug=False):
    H, W = image.shape[:2]

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input = preprocess(image).unsqueeze(0)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()

    if torch.cuda.is_available():
        input = input.to('cuda')
        model.to('cuda')
    
    with torch.no_grad():
        output = model(input)['out'][0].cpu().numpy()
        
    out = np.zeros((H, W), dtype=np.uint8)
    out[output.argmax(0) == 15] = 1 # 15 is the index of the human class

    if debug:
        image = np.where(out[:,:,None] == 1, image, image//8)
        plt.imshow(image)
        plt.show()

    return out

def face_mask(image_path, debug=False):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    out = np.where(face_bound(image, debug=debug),
                   subject_mask(image, debug=debug), 0)
    
    if debug:
        image = np.where(out[:,:,None] == 1, image, image//8)
        plt.imshow(image)
        plt.show()

    return out