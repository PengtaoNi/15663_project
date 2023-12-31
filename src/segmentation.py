import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
from torchvision import transforms
from mtcnn import MTCNN

def get_face_bounds(image, debug=False):
    H, W = image.shape[:2]

    detector = MTCNN()
    faces = detector.detect_faces(image)

    out = []
    for face in faces:
        x, y, w, h = face['box']

        # extend the bounds to cover hair and chin
        x = max(x-w//2, 0)
        w = min(w*2, W-x)
        y = max(y-h, 0)
        h = min(int(h*2.1), H-y)

        mask = np.zeros((H, W), dtype=np.uint8)
        mask[y:y+h, x:x+w] = 1
        out.append(mask)
    
    out = np.stack(out, axis=0)

    if debug:
        for k in range(len(out)):
            plt.imshow(np.where(out[k][:,:,None] == 1, image, image//8))
            plt.show()

    return out

def get_subject_mask(image, debug=False):
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

def get_face_masks(image, debug=False):
    face_bounds = get_face_bounds(image, debug=debug)
    subject_mask = get_subject_mask(image, debug=debug)

    out = []
    for k in range(len(face_bounds)):
        face_mask = np.where(face_bounds[k], subject_mask, 0) # intersection
        out.append(face_mask)
    
    out = np.stack(out, axis=0)
    
    if debug:
        for k in range(len(out)):
            plt.imshow(np.where(out[k][:,:,None] == 1, image, image//8))
            plt.show()

    return out

def get_face_weights(face_masks, dim=(103, 78), padding=4):
    n_faces, H, W = face_masks.shape
    if H < W:
        dim = (dim[1], dim[0])
    h, w = dim

    face_weights = []
    for k in range(n_faces):
        face_weight = cv2.resize(face_masks[k], (w, h))
        face_weights.append(np.pad(face_weight, padding))
    face_weights = np.stack(face_weights, axis=0)

    return face_weights

def get_body_weight(image, dim=(103, 78), padding=4):
    subject_mask = get_subject_mask(image)
    H, W = subject_mask.shape
    if H < W:
        dim = (dim[1], dim[0])
    h, w = dim

    body_weight = cv2.resize(subject_mask, (w, h))
    body_weight = np.pad(body_weight, padding)

    return body_weight[None, :, :]