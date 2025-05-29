# Show Kernel Spectrum of 1st Layer Deep Learning Models

A PyQt5-based visualization tool for analyzing the first convolutional layer weights of various deep learning models through multiple visualization modes including Fourier amplitude spectrum analysis.

## Screenshots

### Application Interface
![Application Interface](ScrShot%2012.png)

### Fourier Spectrum Visualization
![Fourier Spectrum Analysis](ScrShot%2013.png)

## Features

### Supported Models
- **AlexNet** - Classic CNN architecture
- **VGG16** - Deep convolutional network
- **ResNet50** - Residual learning network
- **LeNet5** - Classic handwritten digit recognition network
- **DenseNet121** - Densely connected network
- **MobileNetV2** - Efficient mobile architecture
- **GoogLeNet** - Inception-based architecture
- **InceptionV3** - Advanced Inception architecture

### Visualization Modes
1. **실제 가중치 (Actual Weights)** - Direct visualization of filter weights
2. **고대비 가중치 (High Contrast Weights)** - Histogram equalized weights for enhanced pattern visibility
3. **컬러 채널 시각화 (Color Channel Visualization)** - RGB channel visualization of filters
4. **푸리에 진폭 스펙트럼 (Fourier Amplitude Spectrum)** - 2D FFT analysis of kernel weights

## Key Technical Features

### Fourier Spectrum Analysis
- Kernel resizing from original size (e.g., 11×11 for AlexNet) to 16×16 using smooth interpolation
- 2D Fast Fourier Transform (FFT) with frequency shifting for centered spectrum
- Logarithmic scaling for dynamic range compression
- Real-time spectrum visualization of up to 64 filters

### Advanced Image Processing
- **Smooth Kernel Resizing**: Uses 3rd order spline interpolation via `scipy.ndimage.zoom`
- **Multi-channel Support**: Handles both grayscale and RGB input channels
- **Dynamic Scaling**: Automatic normalization and scaling for optimal visualization
- **Edge Padding**: Intelligent padding for consistent display sizes

## Installation

### Requirements
```bash
pip install torch torchvision
pip install PyQt5
pip install scipy
pip install numpy
```

### Hardware Support
- **Apple Silicon (M1/M2)**: Optimized for MPS (Metal Performance Shaders)
- **NVIDIA GPU**: CUDA support
- **CPU**: Fallback support

## Usage

### Running the Application
```bash
python kernel_spectrum_visualizer.py
```

### Interface Controls
1. **Model Selection**: Choose from 8 different pre-trained models
2. **Visualization Mode**: Select visualization method from dropdown
3. **Real-time Updates**: Automatic visualization updates when changing models or modes

### Visualization Modes Explained

#### 1. Fourier Amplitude Spectrum Mode
This mode performs the following operations:
- Resizes each kernel from its original dimensions to 16×16 using smooth interpolation
- Applies 2D FFT to analyze frequency components
- Centers the spectrum using `fftshift`
- Computes magnitude spectrum with logarithmic scaling
- Displays the result as grayscale intensity maps

**Use Cases:**
- Understanding frequency response characteristics of learned filters
- Analyzing edge detection capabilities
- Comparing spectral properties across different architectures

#### 2. Color Channel Visualization
- Displays RGB channels of filters as colored images
- Handles grayscale inputs by replicating channels
- Provides intuitive understanding of multi-channel filters

#### 3. High Contrast Weights
- Applies histogram equalization for enhanced pattern visibility
- Useful for detecting subtle patterns in weight distributions
- Maximizes dynamic range utilization

## Technical Implementation

### Model Weight Extraction
```python
# Example for AlexNet
model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
weights = model.features[0].weight  # Shape: [64, 3, 11, 11]
```

### Fourier Transform Pipeline
```python
def compute_fft_magnitude(self, input_img):
    # 1. Resize kernel to 16x16
    resized_img = self.resize_kernel_smooth(input_img, target_size=(16, 16))
    
    # 2. Apply 2D FFT
    fft_result = np.fft.fft2(resized_img)
    fft_shifted = np.fft.fftshift(fft_result)
    
    # 3. Compute magnitude spectrum
    magnitude = np.abs(fft_shifted)
    magnitude = np.log1p(magnitude)  # Log scaling
    
    return magnitude
```

### Smooth Kernel Resizing
```python
def resize_kernel_smooth(self, input_img, target_size=(16, 16)):
    zoom_h = target_size[0] / input_img.shape[0]
    zoom_w = target_size[1] / input_img.shape[1]
    return zoom(input_img, (zoom_h, zoom_w), order=3)  # Cubic spline interpolation
```

## Model Architecture Details

| Model | First Layer | Kernel Size | Filters | Input Channels |
|-------|-------------|-------------|---------|----------------|
| AlexNet | `features[0]` | 11×11 | 64 | 3 |
| VGG16 | `features[0]` | 3×3 | 64 | 3 |
| ResNet50 | `conv1` | 7×7 | 64 | 3 |
| LeNet5 | `conv1` | 5×5 | 6 | 1→3* |
| DenseNet121 | `features.conv0` | 7×7 | 64 | 3 |
| MobileNetV2 | `features[0][0]` | 3×3 | 32 | 3 |
| GoogLeNet | `conv1.conv` | 7×7 | 64 | 3 |
| InceptionV3 | `Conv2d_1a_3x3.conv` | 3×3 | 32 | 3 |

*LeNet5 input channels are expanded from 1 to 3 for visualization purposes.

## Frequency Analysis Insights

The Fourier spectrum visualization reveals:
- **Low-frequency components**: Smooth variations and gradients
- **High-frequency components**: Edge detection and texture analysis
- **Directional patterns**: Oriented edge detectors
- **Spatial frequency preferences**: Model-specific frequency biases

## Applications

### Research Applications
- **Filter Analysis**: Understanding what features each filter detects
- **Architecture Comparison**: Comparing frequency responses across models
- **Transfer Learning**: Analyzing pre-trained filter characteristics
- **Model Interpretability**: Visualizing learned representations

### Educational Applications
- **CNN Visualization**: Teaching convolutional neural network concepts
- **Signal Processing**: Demonstrating 2D Fourier analysis
- **Computer Vision**: Understanding image filtering operations

## File Structure
```
.
├── kernel_spectrum_visualizer.py  # Main application
├── README.md                      # This file
├── ScrShot 12.png                # Application interface screenshot
├── ScrShot 13.png                # Fourier spectrum visualization screenshot
└── requirements.txt              # Python dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python kernel_spectrum_visualizer.py`

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- PyTorch team for the deep learning framework
- torchvision for pre-trained models
- PyQt5 for the GUI framework
- SciPy for advanced image processing functions

## Citation

If you use this tool in your research, please consider citing:

```bibtex
@software{kernel_spectrum_visualizer,
  title={Show Kernel Spectrum of 1st Layer Deep Learning Models},
  author={bemoregt},
  year={2025},
  url={https://github.com/bemoregt/Show_KernelSpectrum_1stLayerOfAlexnet}
}
```

---

**Note**: This tool is designed for educational and research purposes. The visualizations provide insights into the learned representations of various CNN architectures through their first convolutional layer weights.