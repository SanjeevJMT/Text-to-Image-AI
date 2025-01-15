import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import warnings
from tqdm import tqdm

class ImageGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", output_dir="generated_images"):
        """
        Initialize the image generator optimized for CPU usage.
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print("Loading model (this may take a few minutes on CPU)...")
        
        # Initialize pipeline with CPU optimizations
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # Use float32 for CPU
            safety_checker=None  # Disable safety checker for better performance
        )
        
        # Memory optimization
        self.pipe.enable_attention_slicing()
        
        # Warning if running on CPU
        if not torch.cuda.is_available():
            warnings.warn(
                "Running on CPU. Generation will be significantly slower. "
                "Consider reducing num_inference_steps for faster generation."
            )
    
    def generate_image(self, prompt, filename=None, num_inference_steps=30, 
                      guidance_scale=7.5, negative_prompt=None):
        """
        Generate an image based on the input prompt, optimized for CPU usage.
        """
        try:
            print(f"Generating image for prompt: '{prompt}'")
            print(f"This may take several minutes on CPU...")
            
            # Show progress bar during generation
            with tqdm(total=num_inference_steps, desc="Generating") as pbar:
                def callback(step, timestep, latents):
                    pbar.update(1)
                
                # Generate the image
                image = self.pipe(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    callback=callback,
                    callback_steps=1
                ).images[0]
            
            # Create filename if not provided
            if filename is None:
                filename = "".join(x for x in prompt if x.isalnum() or x in (' ','-','_'))[:50]
                filename = filename.replace(' ', '_') + '.png'
            
            # Ensure filename has .png extension
            if not filename.endswith('.png'):
                filename += '.png'
            
            # Save the image
            filepath = os.path.join(self.output_dir, filename)
            image.save(filepath)
            
            print(f"Image saved successfully at: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            return None
    
    def generate_image_variations(self, prompt, num_images=2, **kwargs):
        """
        Generate multiple variations of images for the same prompt.
        Reduced default number of variations for CPU usage.
        """
        paths = []
        for i in range(num_images):
            print(f"\nGenerating variation {i+1}/{num_images}")
            filename = f"{i+1}_{kwargs.get('filename', prompt)}"
            path = self.generate_image(prompt, filename=filename, **kwargs)
            if path:
                paths.append(path)
        return paths

def main():
    # Example usage with CPU-optimized parameters
    generator = ImageGenerator()
    
    prompt = "A serene mountain landscape at sunset with a lake reflection"
    
    # Single image generation with reduced steps for CPU
    print("\nGenerating single image...")
    image_path = generator.generate_image(
        prompt=prompt,
        num_inference_steps=20,  # Reduced steps for CPU
        guidance_scale=7.0
    )
    
    if image_path:
        print("\nSingle image generated successfully!")
    
    # Generate fewer variations with CPU-optimized parameters
    print("\nGenerating variations...")
    variation_paths = generator.generate_image_variations(
        prompt=prompt,
        num_images=2,  # Reduced number of variations
        num_inference_steps=20,  # Reduced steps for CPU
        guidance_scale=7.0
    )
    
    print(f"\nGenerated {len(variation_paths)} variations")

if __name__ == "__main__":
    main()