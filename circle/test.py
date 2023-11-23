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

def mixed_test(model):
    
    nc = NoisyCircles(f"{DIR}/test/images.npy", f"{DIR}/test/labels.npy")
    results = []
    for idx in range(len(nc)):
        img, cp = nc[idx]
        cp_hat = find_circle(model, img)
        results.append(iou(CircleParams(*cp), CircleParams(*cp_hat)))
    results = np.array(results)
    
    print(f"Mixed Noise Test: {round(results.mean(), 4)}")

def level_test(model):
    for nl in range(0, 10):
        nl /= 10
        nc = [next(generate_examples(nl, min_radius=5, max_radius=50)) for _ in range(1000)]
        results = []
        for idx in range(len(nc)):
            img, cp = nc[idx]
            cp_hat = find_circle(model, img)
            results.append(iou(cp, CircleParams(*cp_hat)))
        results = np.array(results)
        print(f"{round(nl, 2)} Noise Test: {round(results.mean(), 4)}")

def main():
    
    model = ConvNet()
    checkpoint = torch.load('model.pth.tar', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    
    mixed_test(model)
    print()
    level_test(model)

if __name__ == "__main__":
    main()