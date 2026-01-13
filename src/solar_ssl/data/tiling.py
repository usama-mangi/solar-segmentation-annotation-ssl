import os
from PIL import Image
from tqdm import tqdm

def tile_dataset(input_dir, output_dir, tile_size=256, stride=256):
    """
    Tiles images exactly as defined in tile-images.ipynb.
    Only tiles if the full tile_size fits (no padding).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .png files
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    total_tiles_saved = 0
    
    print(f"Starting tiling process...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Tile size: {tile_size}x{tile_size}, Stride: {stride}")

    for filename in tqdm(image_files, desc="Processing Images"):
        image_path = os.path.join(input_dir, filename)
        
        try:
            # Open the image
            img = Image.open(image_path).convert("RGB")
            img_width, img_height = img.size

            # Get the base filename without extension
            base_filename = os.path.splitext(filename)[0]

            # Slide a window across the image
            # Logic preserved: stops if the remaining space is < TILE_SIZE
            for y in range(0, img_height - tile_size + 1, stride):
                for x in range(0, img_width - tile_size + 1, stride):
                    
                    # Define the crop box (left, upper, right, lower)
                    box = (x, y, x + tile_size, y + tile_size)
                    
                    # Crop the tile
                    tile = img.crop(box)
                    
                    # Create unique name: "original_name_tile_y_x.png"
                    tile_filename = f"{base_filename}_tile_{y}_{x}.png"
                    
                    # Save the tile
                    output_path = os.path.join(output_dir, tile_filename)
                    tile.save(output_path)
                    total_tiles_saved += 1

        except Exception as e:
            print(f"\n[Error] Could not process {filename}: {e}")

    print(f"\n--- Tiling Complete ---")
    print(f"Processed {len(image_files)} images.")
    print(f"Saved {total_tiles_saved} tiles to {output_dir}")