# TextToImageAI

A Python-based text-to-image generation tool using Stable Diffusion, designed for easy integration and customization.

## Description

TextToImageAI is a powerful yet simple-to-use text-to-image generation tool that leverages the Stable Diffusion model to create high-quality images from text descriptions. It supports both single image generation and multiple variations, with customizable parameters for fine-tuning the generation process.

## Features

- 🖼️ Generate high-quality images from text descriptions
- 🎨 Create multiple variations of the same prompt
- 🎯 Customize generation parameters (steps, guidance scale, etc.)
- 🚀 GPU acceleration support for faster generation
- 📁 Automatic file management and organization
- ❌ Negative prompt support for better control
- 🛠️ Error handling and robust implementation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SanjeevJMT/TextToImageAI.git
cd TextToImageAI
```

2. Install the required dependencies:
```bash
pip install torch diffusers transformers pillow
```

## Usage

### Basic Usage

```python
from image_generator import ImageGenerator

# Initialize the generator
generator = ImageGenerator()

# Generate a single image
image_path = generator.generate_image(
    prompt="A serene mountain landscape at sunset with a lake reflection"
)
print(f"Image saved to: {image_path}")
```

### Advanced Usage

```python
# Generate an image with custom parameters
image_path = generator.generate_image(
    prompt="A beautiful sunset over mountains",
    filename="sunset_mountains.png",
    num_inference_steps=50,
    guidance_scale=7.5,
    negative_prompt="blur, distortion, low quality"
)

# Generate multiple variations
variations = generator.generate_image_variations(
    prompt="A cosmic nebula with vibrant colors",
    num_images=4,
    num_inference_steps=30
)
```

## Configuration

The `ImageGenerator` class accepts several parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| model_id | Stable Diffusion model to use | "runwayml/stable-diffusion-v1-5" |
| output_dir | Directory for saving generated images | "generated_images" |
| num_inference_steps | Number of denoising steps | 50 |
| guidance_scale | How closely to follow the prompt | 7.5 |

## System Requirements

- Python 3.7 or higher
- CUDA-capable GPU (recommended)
- Minimum 8GB RAM (16GB recommended)
- ~4GB disk space for model storage

## Performance Notes

- First-time usage will download the model (~4GB)
- GPU acceleration significantly improves generation speed
- Higher inference steps produce better quality but take longer
- Guidance scale affects how closely the image follows the prompt

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is Not Licensed yet.
## Acknowledgments

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) for the base model
- [Hugging Face](https://huggingface.co/) for model hosting and diffusers library

## Contact

Sanjeev JMT - [@SanjeevJMT](https://github.com/SanjeevJMT)

Project Link: [https://github.com/SanjeevJMT/TextToImageAI](https://github.com/SanjeevJMT/TextToImageAI)