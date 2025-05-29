# Show Kernel Spectrum of 1st Layer Deep Learning Models

A PyQt5-based visualization tool for analyzing the first convolutional layer weights of various deep learning models through multiple visualization modes including Fourier amplitude spectrum analysis with intelligent clustering capabilities.

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
4. **푸리에 진폭 스펙트럼 (Fourier Amplitude Spectrum)** - 2D FFT analysis of kernel weights with intelligent clustering

## Key Technical Features

### Advanced Fourier Spectrum Analysis
- **Multi-stage Kernel Processing**: Kernel resizing from original size (e.g., 11×11 for AlexNet) to 16×16, then spectrum upsampling to 32×32
- **2D Fast Fourier Transform (FFT)** with frequency shifting for centered spectrum
- **Logarithmic scaling** for dynamic range compression
- **Real-time spectrum visualization** of up to 64 filters simultaneously

### Intelligent Spectrum Clustering
- **Cosine Similarity Analysis**: Computes similarity between all spectrum pairs
- **DBSCAN Clustering**: Groups similar spectra using density-based clustering algorithm
- **Visual Cluster Identification**: Color-coded borders for filters belonging to the same cluster
- **Adjustable Similarity Threshold**: Fine-tunable clustering sensitivity (default: 0.93)
- **Fallback Clustering**: Robust threshold-based clustering when DBSCAN fails

### Advanced Image Processing
- **Smooth Kernel Resizing**: Uses 3rd order spline interpolation via `scipy.ndimage.zoom`
- **Dual-stage Spectrum Processing**: 16×16 FFT followed by 32×32 interpolation for enhanced detail
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
pip install scikit-learn  # For clustering algorithms
pip install matplotlib    # For color processing
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
4. **Cluster Analysis**: Automatic clustering in Fourier spectrum mode with console output

### Visualization Modes Explained

#### 1. Fourier Amplitude Spectrum Mode (Enhanced)
This advanced mode performs the following operations:
- Resizes each kernel from its original dimensions to 16×16 using smooth interpolation
- Applies 2D FFT to analyze frequency components
- Centers the spectrum using `fftshift`
- Upsamples spectrum to 32×32 for enhanced detail
- Computes magnitude spectrum with logarithmic scaling
- **Performs cosine similarity clustering** on flattened 32×32 spectra
- **Displays color-coded borders** for filters in the same cluster
- Outputs cluster information to console

**Advanced Clustering Features:**
- **Similarity Threshold**: Adjustable cosine similarity threshold (0.93 default)
- **Automatic Grouping**: Identifies filters with similar frequency characteristics
- **Visual Feedback**: 10 distinct colors for different clusters
- **Console Analytics**: Detailed cluster membership information

#### 2. Color Channel Visualization
- Displays RGB channels of filters as colored images
- Handles grayscale inputs by replicating channels
- Provides intuitive understanding of multi-channel filters

#### 3. High Contrast Weights
- Applies histogram equalization for enhanced pattern visibility
- Useful for detecting subtle patterns in weight distributions
- Maximizes dynamic range utilization

## Technical Implementation

### Enhanced Fourier Transform Pipeline
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

def resize_spectrum_to_32x32(self, spectrum):
    # Enhanced spectrum resolution for detailed analysis
    current_h, current_w = spectrum.shape
    zoom_h = 32 / current_h
    zoom_w = 32 / current_w
    
    resized_spectrum = zoom(spectrum, (zoom_h, zoom_w), order=3)
    return resized_spectrum
```

### Intelligent Clustering System
```python
def compute_cosine_similarity_clustering(self, spectrums, threshold=0.93):
    # Flatten 32x32 spectra for similarity analysis
    flattened_spectrums = [spectrum.flatten() for spectrum in spectrums]
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(flattened_spectrums)
    distance_matrix = 1 - similarity_matrix
    distance_matrix = np.clip(distance_matrix, 0, 2)
    
    # DBSCAN clustering with fallback
    eps = 1 - threshold
    try:
        dbscan = DBSCAN(eps=eps, min_samples=2, metric='precomputed')
        cluster_labels = dbscan.fit_predict(distance_matrix)
    except Exception:
        cluster_labels = self.simple_threshold_clustering(similarity_matrix, threshold)
    
    return cluster_labels
```

### Visual Cluster Representation
```python
def draw_border_on_pixmap(self, pixmap, color, border_width=3):
    # Creates colored borders for cluster visualization
    bordered_pixmap = QPixmap(pixmap.size())
    painter = QPainter(bordered_pixmap)
    painter.drawPixmap(0, 0, pixmap)
    
    pen = QPen(QColor(color[0], color[1], color[2]))
    pen.setWidth(border_width)
    painter.setPen(pen)
    painter.drawRect(bordered_pixmap.rect().adjusted(1, 1, -1, -1))
    
    return bordered_pixmap
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

## Advanced Analysis Capabilities

### Spectral Clustering Insights
The intelligent clustering system reveals:
- **Functional Filter Groups**: Identifies filters with similar frequency responses
- **Architectural Patterns**: Discovers repeated filter designs within models
- **Redundancy Analysis**: Highlights potentially redundant filters
- **Transfer Learning Insights**: Understanding of feature hierarchy

### Frequency Analysis Insights
The enhanced Fourier spectrum visualization reveals:
- **Low-frequency components**: Smooth variations and gradients
- **High-frequency components**: Edge detection and texture analysis
- **Directional patterns**: Oriented edge detectors
- **Spatial frequency preferences**: Model-specific frequency biases
- **Cluster Relationships**: Similar filters grouped by spectral characteristics

## Applications

### Research Applications
- **Filter Analysis**: Understanding what features each filter detects
- **Architecture Comparison**: Comparing frequency responses across models
- **Transfer Learning**: Analyzing pre-trained filter characteristics
- **Model Interpretability**: Visualizing learned representations
- **Redundancy Detection**: Identifying similar filters for model compression
- **Spectral Clustering**: Grouping filters by frequency characteristics

### Educational Applications
- **CNN Visualization**: Teaching convolutional neural network concepts
- **Signal Processing**: Demonstrating 2D Fourier analysis
- **Computer Vision**: Understanding image filtering operations
- **Machine Learning**: Illustrating clustering algorithms in practice

## File Structure
```
.
├── kernel_spectrum_visualizer.py  # Main application
├── README.md                      # This file
├── ScrShot 12.png                # Application interface screenshot
├── ScrShot 13.png                # Fourier spectrum visualization screenshot
└── requirements.txt              # Python dependencies
```

## Console Output Example
When running in Fourier Spectrum mode, the application provides detailed clustering information:
```
사용 장치: mps
발견된 클러스터 수: 5
클러스터 0: 필터 [ 2  5 12 18 23]
클러스터 1: 필터 [ 1  8 15 22]
클러스터 2: 필터 [ 3  9 14 19 25]
클러스터 3: 필터 [ 6 11 17 24]
클러스터 4: 필터 [ 4  7 13 16 20 21]
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
- scikit-learn for clustering algorithms

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

**Note**: This tool is designed for educational and research purposes. The visualizations and clustering analysis provide insights into the learned representations of various CNN architectures through their first convolutional layer weights.