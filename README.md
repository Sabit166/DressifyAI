# 👗 DressifyAI

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Streamlit-1.48.1-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/AI-Powered-green.svg" alt="AI Powered">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</div>

**An AI-powered virtual clothing try-on application that combines computer vision and generative AI to transform fashion experiences.**

DressifyAI enables users to seamlessly try on different clothing items by leveraging state-of-the-art object detection, segmentation, and image generation technologies.

## ✨ Features

🎯 **Smart Object Detection**: Automatically detects clothing items in uploaded images using YOLOS (You Only Look at One Sequence) model  
🎨 **Precise Segmentation**: Creates accurate masks using Segment Anything Model (SAM) 2.1  
🤖 **AI Image Generation**: Generates realistic clothing transformations using Stable Diffusion XL Inpainting  
⚡ **Real-time Processing**: Interactive Streamlit interface for immediate results  
📱 **User-Friendly**: Simple upload-and-transform workflow  

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Object Detection**: YOLOv5-based Fashion Object Detection Model
- **Segmentation**: Segment Anything Model (SAM) 2.1
- **Image Generation**: Stable Diffusion XL Inpainting API (Segmind)
- **Computer Vision**: OpenCV, PIL
- **Deep Learning**: PyTorch, Transformers, Ultralytics

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sabit166/DressifyAI.git
   cd DressifyAI
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv env
   # Windows
   .\env\Scripts\activate
   # macOS/Linux
   source env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   API_KEY=your_segmind_api_key_here
   URL=https://api.segmind.com/v1/sdxl-inpaint
   ```

5. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

6. **Open your browser** to `http://localhost:8501`

## 📋 How It Works

1. **Upload Image**: Start by uploading a photo of a person wearing clothes
2. **Enter Prompt**: Describe the clothing transformation you want (e.g., "A red dress", "Blue jeans")
3. **Object Detection**: The AI automatically detects clothing items in the image
4. **Generate Mask**: Create a segmentation mask for the detected clothing
5. **Transform**: Generate a new image with the requested clothing change

## 📁 Project Structure

```
DressifyAI/
├── streamlit_app.py           # Main Streamlit application
├── requirements.txt           # Python dependencies
├── .env                      # Environment variables (create this)
├── models/                   # AI model implementations
│   ├── detector.py          # YOLOS object detection
│   ├── segmentation_mask.py # SAM segmentation
│   ├── image_generation.py  # Stable Diffusion inpainting
│   └── test_env_variables.py # Environment testing
├── sam2.1_b.pt              # SAM model weights
├── yolov8n.pt               # YOLO model weights
└── README.md                # This file
```

## 🎮 Usage Examples

### Basic Clothing Transformation
```python
# Example prompts for image generation:
"A elegant black dress"
"Casual blue jeans and white t-shirt"
"Professional business suit"
"Summer floral dress"
"Vintage denim jacket"
```

## 🔧 Configuration

### API Setup
The application uses Segmind's Stable Diffusion XL Inpainting API. You'll need to:

1. Sign up at [Segmind](https://www.segmind.com/)
2. Get your API key
3. Add it to your `.env` file

### Model Weights
- SAM model (`sam2.1_b.pt`) is automatically downloaded on first run
- YOLO model (`yolov8n.pt`) is included in the repository

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8 and SAM implementations
- [Meta AI](https://github.com/facebookresearch/segment-anything) for Segment Anything Model
- [Hugging Face](https://huggingface.co/) for the fashion object detection model
- [Segmind](https://www.segmind.com/) for the Stable Diffusion XL API
- [Streamlit](https://streamlit.io/) for the amazing web app framework

## 📧 Contact

**Sabit Hasan** - [@Sabit166](https://github.com/Sabit166)

Project Link: [https://github.com/Sabit166/DressifyAI](https://github.com/Sabit166/DressifyAI)

---

<div align="center">
  <p>Made with ❤️ and AI</p>
  <p>⭐ Star this repo if you found it helpful!</p>
</div>
