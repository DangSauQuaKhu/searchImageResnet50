from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np
import os

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    fe = FeatureExtractor()

    for img_path in sorted(Path("./static/ImageSearch").glob("*.jpg")):
        print(img_path)  # e.g., ./static/img/xxx.jpg
        feature = fe.extract(img=Image.open(img_path))
        feature_path = Path("./static/feature") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)