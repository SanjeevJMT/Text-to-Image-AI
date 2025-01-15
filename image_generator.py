import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

class ImageGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", output_dir="generated_images"):
        """
        Initialize the image generator.
        
        Args:
            model_id (str): The model ID to use from Hugging Face
            output_dir (str): Directory to save generated images
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Initialize the pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")
    
    def generate_image(self, prompt, filename=None, num_inference_steps=50, 
                      guidance_scale=7.5, negative_prompt=None):
        """
        Generate an image based on the input prompt.
        
        Args:
            prompt (str): The text prompt to generate an image from
            filename (str, optional): Filename to save the image. If None, will use sanitized prompt
            num_inference_steps (int): Number of denoising steps (higher = better quality but slower)
            guidance_scale (float): How closely to follow the prompt (higher = closer but may be less natural)
            negative_prompt (str, optional): Things to avoid in the image
            
        Returns:
            str: Path to the saved image
        """
        try:
            # Generate the image
            image = self.pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt
            ).images[0]
            
            # Create filename if not provided
            if filename is None:
                # Sanitize prompt for filename
                filename = "".join(x for x in prompt if x.isalnum() or x in (' ','-','_'))[:50]
                filename = filename.replace(' ', '_') + '.png'
            
            # Ensure filename has .png extension
            if not filename.endswith('.png'):
                filename += '.png'
            
            # Save the image
            filepath = os.path.join(self.output_dir, filename)
            image.save(filepath)
            
            return filepath
            
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            return None
    
    def generate_image_variations(self, prompt, num_images=4, **kwargs):
        """
        Generate multiple variations of images for the same prompt.
        
        Args:
            prompt (str): The text prompt to generate images from
            num_images (int): Number of variations to generate
            **kwargs: Additional arguments to pass to generate_image
            
        Returns:
            list: List of paths to the saved images
        """
        paths = []
        for i in range(num_images):
            filename = f"{i+1}_{kwargs.get('filename', prompt)}"
            path = self.generate_image(prompt, filename=filename, **kwargs)
            if path:
                paths.append(path)
        return paths

# Example usage
def main():
    # Initialize the generator
    generator = ImageGenerator()
    
    # Single image generation
    prompt = "A serene mountain landscape at sunset with a lake reflection"
    image_path = generator.generate_image(
        prompt=prompt,
        negative_prompt="blur, distortion, low quality",
        num_inference_steps=30,
        guidance_scale=7.5
    )
    
    if image_path:
        print(f"Image generated successfully at: {image_path}")
    
    # Generate multiple variations
    variation_paths = generator.generate_image_variations(
        prompt=prompt,
        num_images=3,
        num_inference_steps=30,
        guidance_scale=7.5
    )
    
    print(f"Generated {len(variation_paths)} variations")

if __name__ == "__main__":
    main()