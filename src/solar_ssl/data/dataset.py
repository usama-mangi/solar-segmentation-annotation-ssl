import os
import random
from PIL import Image
from torch.utils.data import Dataset

class SatelliteImageDataset(Dataset):
    """
    Loads the FULL satellite images.
    The SimCLR transform will handle the random cropping (tiling).
    Ref: encoder-moco.ipynb
    """
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        # Just get a list of all image paths
        self.image_files = []
        for f in os.listdir(image_dir):
            if f.endswith(('.png', '.jpg', '.jpeg')):
                self.image_files.append(os.path.join(image_dir, f))
                
        self.transform = transform
        print(f"Dataset created with {len(self.image_files)} images.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        
        try:
            # Open the full image
            image = Image.open(image_path).convert("RGB")
            
            # Apply the SimCLR transform (which returns view_1, view_2)
            return self.transform(image)
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Fallback: get a different random image
            random_idx = random.randint(0, len(self)-1)
            return self.__getitem__(random_idx)