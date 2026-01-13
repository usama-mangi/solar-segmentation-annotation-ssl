from torchvision import transforms

class SimCLRDataTransform:
    """
    Yeh transform class ek image leti hai aur uske do augmented versions (views) return karti hai.
    Ref: encoder-moco.ipynb
    """
    def __init__(self, image_size=128):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=image_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # Color jitter SimCLR ke liye bahut zaroori hai
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image):
        # Ek hi image par do baar alag-alag transform apply karein
        view_1 = self.transform(image)
        view_2 = self.transform(image)
        return view_1, view_2