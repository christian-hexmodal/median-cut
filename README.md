# Median-Cut Color Quantization

A powerful web-based tool for image color quantization using the median-cut algorithm. This application provides an interactive interface to reduce the number of colors in images while maintaining visual quality, with comprehensive analysis and export capabilities.

## 🎨 Features

- **Interactive Color Quantization**: Reduce images to 2, 4, 8, 16, 32, or 64 colors
- **Real-time Comparison**: Side-by-side comparison of original and quantized images
- **Quality Metrics**: Comprehensive analysis including MSE, PSNR, SSIM, and compression ratios
- **Color Palette Analysis**: Detailed color palette visualization with pixel count histograms
- **Export Options**: Download quantized images, color palettes, and detailed reports
- **Undo/Redo**: Session history management for workflow flexibility
- **Color Similarity Highlighting**: Advanced color analysis with similarity detection

## 🚀 Access the Application

The Median-Cut Color Quantization tool is available as a public web application:

**🌐 Live Demo: [https://mediancut.streamlit.app/](https://mediancut.streamlit.app/)**

Simply visit the link above to start using the application immediately - no installation required!

## 📦 Dependencies

For developers who want to run the code locally, the application requires the following Python packages (see `requirements.txt` for specific versions):

- **Streamlit**: Web application framework
- **NumPy**: Numerical computing and array operations
- **Pillow (PIL)**: Image processing and manipulation
- **OpenCV (cv2)**: Computer vision and image analysis
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive plotting and visualization (optional)

**Note**: End users can access the application directly at [https://mediancut.streamlit.app/](https://mediancut.streamlit.app/) without installing anything.

## 🎯 Usage

### Basic Workflow

1. **Upload an Image**: Use the file uploader to select a JPG, JPEG, or PNG image
2. **View Comparison**: The app automatically generates quantized versions with 2, 4, 8, 16, 32, and 64 colors
3. **Analyze Results**: Review quality metrics and color palette information
4. **Export**: Download your preferred quantized image and analysis reports

### Advanced Features

#### Color Palette Analysis
- **Color Swatches**: Visual representation of the generated color palette
- **Histogram**: Pixel count distribution across colors
- **Color Highlighting**: Identify similar colors using various similarity metrics

#### Quality Metrics
- **MSE (Mean Squared Error)**: Measures pixel-wise difference between original and quantized
- **PSNR (Peak Signal-to-Noise Ratio)**: Logarithmic measure of image quality
- **SSIM (Structural Similarity Index)**: Perceptual quality assessment
- **Compression Ratio**: File size reduction achieved

#### Export Options
- **Quantized Image**: Download the processed image
- **Color Palette**: Export palette as image or CSS
- **Text Report**: Detailed metrics and analysis
- **HTML Report**: Complete report with embedded comparison images
- **Comparison Image**: Side-by-side visualization

## 🔧 Technical Details

### Median-Cut Algorithm

The median-cut algorithm works by:

1. **Color Space Analysis**: Converting the image to RGB color space
2. **Range Calculation**: Finding the color channel with the largest range
3. **Recursive Splitting**: Dividing the color space into smaller boxes
4. **Color Averaging**: Computing representative colors for each box
5. **Nearest Neighbor Mapping**: Mapping original pixels to the closest palette color

### Algorithm Complexity

- **Time Complexity**: O(n log k) where n is the number of pixels and k is the number of colors
- **Space Complexity**: O(n) for storing pixel data
- **Quality**: Provides good balance between compression and visual quality

## 📊 Performance Metrics

The application calculates several quality metrics:

- **MSE**: Lower values indicate better quality
- **PSNR**: Higher values indicate better quality (typically 20-40 dB for good results)
- **SSIM**: Values range from 0 to 1, with 1 being identical
- **Compression Ratio**: Shows the reduction in color information

## 🎨 Use Cases

- **Web Design**: Create color palettes for websites and applications
- **Image Compression**: Reduce file sizes while maintaining quality
- **Artistic Effects**: Create stylized or retro image effects
- **Data Visualization**: Generate consistent color schemes
- **Print Design**: Optimize images for limited color printing

## 🔍 Troubleshooting

### Common Issues

1. **Image Upload Problems**
   - Ensure the image is in JPG, JPEG, or PNG format
   - Check file size (recommended < 10MB for optimal performance)

2. **Performance Issues**
   - Large images may take longer to process
   - Consider resizing very large images before upload

3. **Export Issues**
   - Ensure sufficient disk space for downloads
   - Check browser download settings

### Browser Compatibility

- **Chrome/Chromium**: Full support
- **Firefox**: Full support
- **Safari**: Full support
- **Edge**: Full support

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit Team**: For the excellent web app framework
- **OpenCV Community**: For computer vision capabilities
- **Pillow Maintainers**: For image processing functionality
- **NumPy Team**: For numerical computing support

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/median-cut/issues) page
2. Create a new issue with detailed information
3. Include your operating system, Python version, and error messages

---

**Happy Color Quantizing! 🎨**
