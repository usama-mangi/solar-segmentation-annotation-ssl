import os
import gradio as gr
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor

class SolarAnnotator:
    def __init__(self, checkpoint_path="weights/sam_vit_h_4b8939.pth", model_type="vit_h"):
        print("Model load ho raha hai...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(self.device)
        self.predictor = SamPredictor(self.sam)
        print("Model load ho gaya hai.")
        
        # State variables (Replacing globals)
        self.selected_image = None
        self.combined_mask = None
        self.original_filename = None

    def apply_mask_to_image(self, image_rgb, mask):
        """Mask ko image par draw karta hai."""
        color = np.array([30, 144, 255]) # Blue color
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        
        # Original image ko wapas BGR karein display ke liye (matches notebook logic)
        output_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Mask blending
        output_image = cv2.addWeighted(output_image, 0.7, mask_image.astype(np.uint8), 0.3, 0)
        
        # Return as BGR (Gradio will display this correctly if type='numpy')
        return output_image

    def store_image_and_init(self, image):
        """Jab user image upload karta hai."""
        if image is None: return None
        
        # Gradio passes image as numpy array (RGB)
        # Note: In notebook, input was BGR via cv2.imread usually, but Gradio image component sends RGB.
        # We ensure consistency here.
        self.selected_image = image 
        
        self.predictor.set_image(self.selected_image)
        self.combined_mask = np.zeros(self.selected_image.shape[:2], dtype=bool)
        print("Image predictor mein set ho gayi hai.")
        return image

    def segment_and_add_mask(self, evt: gr.SelectData):
        """Jab user image par click karta hai."""
        if self.selected_image is None:
            return None

        # Click ke coordinates
        input_point = np.array([[evt.index[0], evt.index[1]]])
        input_label = np.array([1]) # Foreground
        
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        
        new_mask = masks[0]
        # Logic preservation: Accumulate masks
        self.combined_mask = np.logical_or(self.combined_mask, new_mask)
        
        # Convert RGB image to BGR for drawing function, then return
        output_image = self.apply_mask_to_image(self.selected_image, self.combined_mask)
        
        # Convert back to RGB for Gradio display
        return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    def clear_all_masks(self):
        if self.selected_image is None: return None
        self.combined_mask = np.zeros(self.selected_image.shape[:2], dtype=bool)
        print("Masks cleared.")
        return self.selected_image

    def save_image_to_png(self):
        if self.combined_mask is None:
            print("Error: Pehle image upload karein.")
            return None
            
        # Final image BGR banayein
        output_image_bgr = self.apply_mask_to_image(self.selected_image, self.combined_mask)
        save_path = "masked_output.png"
        cv2.imwrite(save_path, output_image_bgr)
        print(f"Image saved to: {save_path}")
        return save_path

def launch_app():
    annotator = SolarAnnotator()
    
    with gr.Blocks() as demo:
        gr.Markdown("# 🤖 SAM Interactive UI (Multiple Masks)")
        
        with gr.Row():
            input_img = gr.Image(label="Image Upload", type="numpy")
            output_img = gr.Image(label="Result", type="numpy", show_label=False)
            
        with gr.Row():
            clear_btn = gr.Button("Clear Masks")
            save_btn = gr.Button("Save Image")
            
        download_file = gr.File(label="Download")

        input_img.upload(annotator.store_image_and_init, [input_img], [output_img])
        output_img.select(annotator.segment_and_add_mask, [], [output_img])
        clear_btn.click(annotator.clear_all_masks, [], [output_img])
        save_btn.click(annotator.save_image_to_png, [], [download_file])
        
    demo.launch(share=True)

if __name__ == "__main__":
    launch_app()