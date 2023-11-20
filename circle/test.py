import torch
import numpy as np
import torchvision.transforms as transforms
from circle.dataset import NoisyCircles
from circle.utils import iou, CircleParams, generate_examples
from circle.regression import ConvNet

DIR = "./data"

def find_circle(model, x):
    with torch.no_grad():
        image = np.expand_dims(np.asarray(x), axis=0)
        image = torch.from_numpy(np.array(image, dtype=np.float32))
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        image = normalize(image)
        image = image.unsqueeze(0)
        output = model(image)
    return [round(i) for i in (x.shape[0]*output).tolist()[0]]

def main():
    
    model = ConvNet()
    checkpoint = torch.load('model.pth.tar')
    model.load_state_dict(checkpoint)
    model.eval()
    
    nc = NoisyCircles(f"{DIR}/test/images.npy", f"{DIR}/test/labels.npy")
    results = []
    for idx in range(len(nc)):
        img, cp = nc[idx]
        cp_hat = find_circle(model, img)
        results.append(iou(CircleParams(*cp), CircleParams(*cp_hat)))
    results = np.array(results)
    print(results.mean()) # 0.9869

if __name__ == "__main__":
    main()