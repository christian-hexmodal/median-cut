import streamlit as st
import numpy as np
from PIL import Image
import cv2
from collections import Counter
import io
import base64
from datetime import datetime

def median_cut(image, depth):
    # Convert image to RGB if it has alpha channel
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    
    # Convert image to numpy array of RGB values
    pixels = np.array(image)
    pixels = pixels.reshape(-1, 3)
    
    def cut_box(pixels, depth):
        if depth == 0:
            # Average color in the box becomes the palette color
            return [np.mean(pixels, axis=0).astype(int)]
        
        # Find channel with largest range
        ranges = np.ptp(pixels, axis=0)
        channel = np.argmax(ranges)
        
        # Sort pixels by the channel value
        pixels = pixels[pixels[:,channel].argsort()]
        
        # Split pixels into two groups
        mid = len(pixels) // 2
        return cut_box(pixels[:mid], depth-1) + cut_box(pixels[mid:], depth-1)

    # Get palette colors
    palette = cut_box(pixels, depth)
    
    # Quantize image using nearest neighbor
    def find_nearest_color(pixel, palette):
        distances = np.sqrt(np.sum((palette - pixel) ** 2, axis=1))
        return palette[np.argmin(distances)]
    
    quantized = np.array([find_nearest_color(p, palette) for p in pixels])
    quantized = quantized.reshape(image.size[1], image.size[0], 3)
    
    return Image.fromarray(quantized.astype('uint8')), palette

def calculate_quality_metrics(original, quantized):
    """Calculate quality metrics between original and quantized images"""
    # Ensure both images are in RGB format
    if original.mode in ('RGBA', 'LA', 'P'):
        original = original.convert('RGB')
    if quantized.mode in ('RGBA', 'LA', 'P'):
        quantized = quantized.convert('RGB')
    
    original_array = np.array(original)
    quantized_array = np.array(quantized)
    
    # Mean Squared Error (MSE)
    mse = np.mean((original_array - quantized_array) ** 2)
    
    # Peak Signal-to-Noise Ratio (PSNR)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # Structural Similarity Index (SSIM) - simplified calculation
    # Convert to grayscale for SSIM calculation
    original_gray = cv2.cvtColor(original_array, cv2.COLOR_RGB2GRAY)
    quantized_gray = cv2.cvtColor(quantized_array, cv2.COLOR_RGB2GRAY)
    
    # Calculate SSIM manually (simplified version)
    mu1 = np.mean(original_gray)
    mu2 = np.mean(quantized_gray)
    sigma1_sq = np.var(original_gray)
    sigma2_sq = np.var(quantized_gray)
    sigma12 = np.mean((original_gray - mu1) * (quantized_gray - mu2))
    
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
    
    # Compression ratio
    original_size = original_array.size
    quantized_size = quantized_array.size
    compression_ratio = original_size / quantized_size
    
    return {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim,
        'compression_ratio': compression_ratio
    }

def get_image_download_link(img, filename, text):
    """Generate a download link for an image"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def generate_palette_css(palette):
    """Generate CSS color codes from palette"""
    css_colors = []
    hex_colors = []
    
    for i, color in enumerate(palette):
        r, g, b = color
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        css_colors.append(f"/* Color {i+1} */\nbackground-color: rgb({r}, {g}, {b});\ncolor: {hex_color};")
        hex_colors.append(hex_color)
    
    return css_colors, hex_colors

# Initialize session state for undo/redo
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_index' not in st.session_state:
    st.session_state.current_index = -1

st.title('Image Color Quantization using Median Cut')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display original image
    with st.spinner("📁 Loading image..."):
        image = Image.open(uploaded_file)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        
        # Show original image as reference
        st.subheader("📸 Original Image")
        st.image(image, use_container_width=True, caption=f"Size: {image.size[0]}×{image.size[1]} pixels")
        
        st.markdown("---")
        
        # Show comparison mode info
        st.info("Showing comparison: Original + 2, 4, 8, 16, 32, 64 colors")
        
        # Undo/Redo buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Undo") and st.session_state.current_index > 0:
                st.session_state.current_index -= 1
                st.rerun()
        
        with col2:
            if st.button("Redo") and st.session_state.current_index < len(st.session_state.history) - 1:
                st.session_state.current_index += 1
                st.rerun()
    
    # Generate quantized images for different color counts
    color_counts = [2, 4, 8, 16, 32, 64]
    quantized_images = []
    palettes = []
    metrics_list = []
    
    # Show loading progress
    with st.spinner("🔄 Processing image quantization..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, num_colors in enumerate(color_counts):
            status_text.text(f"Processing {num_colors} colors...")
            progress_bar.progress((i + 1) / len(color_counts))
            
            depth = int(np.log2(num_colors))
            quantized_img, palette = median_cut(image, depth)
            metrics = calculate_quality_metrics(image, quantized_img)
            
            quantized_images.append(quantized_img)
            palettes.append(palette)
            metrics_list.append(metrics)
        
        status_text.text("✅ Processing complete!")
        progress_bar.empty()
        status_text.empty()
    
    # Save to history (using the 8-color version as representative)
    current_state = {
        'depth': 3,  # 8 colors
        'quantized_image': quantized_images[2],  # 8-color version
        'palette': palettes[2],
        'metrics': metrics_list[2],
        'timestamp': datetime.now()
    }
    
    # Only add to history if it's different from the last state
    if (not st.session_state.history or 
        st.session_state.history[st.session_state.current_index]['depth'] != 3):
        st.session_state.history.append(current_state)
        st.session_state.current_index = len(st.session_state.history) - 1
    
  
    # Multi-level comparison
    st.subheader("Color Quantization Comparison")
    
    # Create a grid layout for all images
    cols = st.columns(7)  # Original + 6 quantized versions
    
    with cols[0]:
        st.caption("Original")
        st.image(image, use_container_width=True)
    
    for i, (num_colors, quantized_img) in enumerate(zip(color_counts, quantized_images)):
        with cols[i+1]:
            st.caption(f"{num_colors} colors")
            st.image(quantized_img, use_container_width=True)
    
    # Quality Metrics Comparison
    st.subheader("Quality Metrics Comparison")
    
    # Create a table of metrics for all versions
    metrics_data = []
    for i, (num_colors, metrics) in enumerate(zip(color_counts, metrics_list)):
        metrics_data.append({
            "Colors": num_colors,
            "MSE": f"{metrics['mse']:.2f}",
            "PSNR": f"{metrics['psnr']:.2f} dB",
            "SSIM": f"{metrics['ssim']:.3f}",
            "Compression": f"{metrics['compression_ratio']:.1f}x"
        })
    
    # Display as a table
    import pandas as pd
    df = pd.DataFrame(metrics_data)
    st.dataframe(df, use_container_width=True)
    
    # Color Palette Comparison
    st.subheader("Color Palette Comparison")
    
    # Let user select which palette to view
    selected_palette_index = st.selectbox(
        "Choose palette to view:",
        options=range(len(color_counts)),
        format_func=lambda x: f"{color_counts[x]} colors",
        index=2  # Default to 8 colors
    )
    
    selected_palette = palettes[selected_palette_index]
    selected_quantized = quantized_images[selected_palette_index]
    selected_color_count = color_counts[selected_palette_index]
    
    # Create a better palette visualization
    palette_viz = np.zeros((60 * len(selected_palette), 80, 3), dtype=np.uint8)
    for i, color in enumerate(selected_palette):
        palette_viz[i*60:(i+1)*60, :] = color
    
    # Create histogram visualization
    def create_palette_histogram(palette, quantized_image):
        """Create a histogram showing pixel count per color"""
        # Convert quantized image to array and flatten
        img_array = np.array(quantized_image)
        pixels = img_array.reshape(-1, 3)
        
        # Count occurrences of each color in the palette
        color_counts = {}
        for i, palette_color in enumerate(palette):
            # Find pixels that match this palette color
            matches = np.all(pixels == palette_color, axis=1)
            count = np.sum(matches)
            color_counts[i] = count
        
        return color_counts
    
    def sort_colors_by_hue(palette):
        """Sort colors by HSV hue value for better visualization"""
        # Convert RGB to HSV for sorting
        hsv_colors = []
        for color in palette:
            # Convert RGB to HSV (0-1 range)
            r, g, b = color[0]/255, color[1]/255, color[2]/255
            max_val = max(r, g, b)
            min_val = min(r, g, b)
            diff = max_val - min_val
            
            if max_val == min_val:
                h = 0  # Grayscale
            elif max_val == r:
                h = (60 * ((g - b) / diff) + 360) % 360
            elif max_val == g:
                h = (60 * ((b - r) / diff) + 120) % 360
            else:  # max_val == b
                h = (60 * ((r - g) / diff) + 240) % 360
            
            hsv_colors.append((h, color))
        
        # Sort by hue and return sorted colors
        hsv_colors.sort(key=lambda x: x[0])
        return [color for _, color in hsv_colors]
    
    # Sort palette by color
    sorted_palette = sort_colors_by_hue(selected_palette)
    hist_data = create_palette_histogram(selected_palette, selected_quantized)
    
    # Display palette with tabs
    tab1, tab2, tab3 = st.tabs(["Color Swatches", "Histogram", "Export"])
    
    with tab1:
        st.write("**Color Swatches (sorted by hue):**")
        # Create a grid layout for color swatches
        cols = st.columns(min(4, len(sorted_palette)))
        for i, color in enumerate(sorted_palette):
            col_idx = i % 4
            with cols[col_idx]:
                # Create a color swatch with hex code
                hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                st.markdown(f"""
                <div style="
                    background-color: {hex_color}; 
                    width: 100%; 
                    height: 40px; 
                    border: 1px solid #ccc; 
                    border-radius: 4px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: {'white' if (color[0]*0.299 + color[1]*0.587 + color[2]*0.114) < 128 else 'black'};
                    font-family: monospace;
                    font-size: 12px;
                ">{hex_color}</div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.write("**Pixel Count per Color:**")
        
        # Show loading for histogram generation
        with st.spinner("📊 Generating histogram..."):
            # Color highlighting feature
            enable_highlighting = st.checkbox("Enable color highlighting", value=False, 
                                            help="Highlight colors similar to a chosen color")
        
        target_rgb = None
                # Define color similarity functions
        def color_distance_euclidean(color1, color2):
            """Calculate Euclidean distance between two RGB colors"""
            return np.sqrt(sum((a - b) ** 2 for a, b in zip(color1, color2)))
        
        def color_similarity_cosine(color1, color2):
            """Calculate cosine similarity between two RGB colors"""
            # Convert to numpy arrays
            c1 = np.array(color1)
            c2 = np.array(color2)
            
            # Calculate cosine similarity
            dot_product = np.dot(c1, c2)
            norm1 = np.linalg.norm(c1)
            norm2 = np.linalg.norm(c2)
            
            if norm1 == 0 or norm2 == 0:
                return 0
            
            return dot_product / (norm1 * norm2)
        
        def color_distance_perceptual(color1, color2):
            """Calculate perceptual color distance using weighted RGB"""
            # Weights for RGB based on human perception (luminance)
            weights = [0.299, 0.587, 0.114]  # R, G, B weights
            
            weighted_diff = sum(w * (a - b) ** 2 for w, a, b in zip(weights, color1, color2))
            return np.sqrt(weighted_diff)
        
        def highlight_similar_colors(colors, target_rgb, method='cosine', threshold=0.8):
            """Highlight colors that are similar to the target color"""
            highlighted = []
            
            for color in colors:
                if method == 'cosine':
                    similarity = color_similarity_cosine(color, target_rgb)
                    is_similar = similarity >= threshold
                elif method == 'perceptual':
                    distance = color_distance_perceptual(color, target_rgb)
                    is_similar = distance <= threshold
                else:  # euclidean
                    distance = color_distance_euclidean(color, target_rgb)
                    is_similar = distance <= threshold
                
                highlighted.append(is_similar)
            
            return highlighted
        
        if enable_highlighting:
            st.write("**Highlight Similar Colors:**")
            target_color = st.color_picker("Choose a color to highlight similar colors:", value="#FFFF00")
            
            # Convert hex to RGB
            target_rgb = tuple(int(target_color[i:i+2], 16) for i in (1, 3, 5))
            
            # Method selection for color similarity
            similarity_method = st.selectbox(
                "Choose similarity method:",
                ["cosine", "perceptual", "euclidean"],
                help="Cosine: Direction-based similarity (0-1, higher=more similar)\nPerceptual: Human vision weighted (lower=more similar)\nEuclidean: Simple RGB distance (lower=more similar)"
            )
            
            # Threshold adjustment based on method
            if similarity_method == 'cosine':
                threshold = st.slider("Similarity threshold (0.0-1.0)", 0.0, 1.0, 0.95, 0.01, 
                                    help="Higher values = more strict matching. 0.95 = very similar, 0.8 = somewhat similar")
            elif similarity_method == 'perceptual':
                threshold = st.slider("Distance threshold (0-50)", 0, 50, 15, 1,
                                    help="Lower values = more strict matching. 5 = very similar, 25 = somewhat similar")
            else:  # euclidean
                threshold = st.slider("Distance threshold (0-150)", 0, 150, 60, 5,
                                    help="Lower values = more strict matching. 30 = very similar, 100 = somewhat similar")
            
            # Highlight similar colors
            highlighted_colors = highlight_similar_colors(sorted_palette, target_rgb, method=similarity_method, threshold=threshold)
        else:
            # No highlighting - all colors are not highlighted
            highlighted_colors = [False] * len(sorted_palette)
        
        # Create histogram showing pixel count per color
        # Prepare data for plotting
        color_labels = []
        pixel_counts = []
        hex_colors = []
        
        for i, color in enumerate(sorted_palette):
            hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            color_labels.append(f"Color {i+1}")
            # Find the index of this color in the selected palette
            original_index = None
            for j, orig_color in enumerate(selected_palette):
                if np.array_equal(color, orig_color):
                    original_index = j
                    break
            pixel_counts.append(hist_data[original_index])
            hex_colors.append(hex_color)
        
        # Sort by pixel count (most used to least used)
        sorted_data = sorted(zip(color_labels, pixel_counts, hex_colors), key=lambda x: x[1], reverse=True)
        color_labels = [item[0] for item in sorted_data]
        pixel_counts = [item[1] for item in sorted_data]
        hex_colors = [item[2] for item in sorted_data]
        
        # Try Plotly first, then matplotlib, then text
        try:
            import plotly.graph_objects as go
            
            # Create bar chart with highlighting
            fig = go.Figure(data=[
                go.Bar(
                    x=color_labels,
                    y=pixel_counts,
                    marker_color=hex_colors,
                    text=[f"{count:,}" for count in pixel_counts],
                    textposition='auto',
                    name='Pixel Count'
                )
            ])
            
            # Add highlighting for similar colors
            for i, (label, count, hex_color, is_highlighted) in enumerate(zip(color_labels, pixel_counts, hex_colors, highlighted_colors)):
                if is_highlighted:
                    fig.add_shape(
                        type="rect",
                        x0=i-0.4, x1=i+0.4,
                        y0=0, y1=count,
                        line=dict(color="red", width=3),
                        fillcolor="rgba(255,0,0,0.1)"
                    )
            
            # Set title based on highlighting status
            title = "Pixel Count per Color"
            if enable_highlighting:
                title += " (Red border = similar to selected color)"
            
            fig.update_layout(
                title=title,
                xaxis_title="Colors",
                yaxis_title="Number of Pixels",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except ImportError:
            # Try matplotlib as fallback
            try:
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(range(len(color_labels)), pixel_counts, color=hex_colors)
                ax.set_xlabel('Colors')
                ax.set_ylabel('Number of Pixels')
                # Set title based on highlighting status
                title = 'Pixel Count per Color'
                if enable_highlighting:
                    title += ' (Red border = similar to selected color)'
                ax.set_title(title)
                ax.set_xticks(range(len(color_labels)))
                ax.set_xticklabels(color_labels, rotation=45)
                
                # Add highlighting for similar colors
                for i, (bar, is_highlighted) in enumerate(zip(bars, highlighted_colors)):
                    if is_highlighted:
                        bar.set_edgecolor('red')
                        bar.set_linewidth(3)
                
                # Add value labels on bars
                for i, (bar, count) in enumerate(zip(bars, pixel_counts)):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(pixel_counts)*0.01,
                           f'{count:,}', ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
            except ImportError:
                # Fallback to simple text display
                st.write("**Pixel Count per Color:**")
                total_pixels = sum(pixel_counts)
                for i, (label, count, hex_color, is_highlighted) in enumerate(zip(color_labels, pixel_counts, hex_colors, highlighted_colors)):
                    percentage = (count / total_pixels) * 100
                    highlight_text = " 🔴 SIMILAR" if is_highlighted else ""
                    st.write(f"{label} ({hex_color}): {count:,} pixels ({percentage:.1f}%){highlight_text}")
        
        # Show summary statistics
        total_pixels = sum(pixel_counts)
        st.write(f"**Summary:** Total pixels: {total_pixels:,}")
        st.write(f"Most used color: {color_labels[pixel_counts.index(max(pixel_counts))]} ({max(pixel_counts):,} pixels, {max(pixel_counts)/total_pixels*100:.1f}%)")
        st.write(f"Least used color: {color_labels[pixel_counts.index(min(pixel_counts))]} ({min(pixel_counts):,} pixels, {min(pixel_counts)/total_pixels*100:.1f}%)")
    
    with tab3:
        # Generate palette export
        with st.spinner("🎨 Preparing palette export..."):
            css_colors, hex_colors = generate_palette_css(palette)
        
        st.write("**Hex Colors:**")
        for i, hex_color in enumerate(hex_colors):
            st.code(hex_color)
        
        # Download palette as CSS
        css_content = "/* Generated Color Palette */\n\n"
        for css_color in css_colors:
            css_content += css_color + "\n\n"
        
        st.download_button(
            label="Download CSS Palette",
            data=css_content,
            file_name=f"palette_{num_colors}_colors.css",
            mime="text/css"
        )
    
    # Export Section
    st.markdown("---")
    st.markdown("### 📤 Export Options")
    
    # Create a nice container for export options
    with st.container():
        st.markdown("**Choose which version to export:**")
        
        # Let user select which version to export
        export_index = st.selectbox(
            "Select version:",
            options=range(len(color_counts)),
            format_func=lambda x: f"{color_counts[x]} colors",
            index=2,  # Default to 8 colors
            help="Choose which quantized version to export"
        )
        
        export_quantized = quantized_images[export_index]
        export_palette = palettes[export_index]
        export_metrics = metrics_list[export_index]
        export_color_count = color_counts[export_index]
        
        # Show selected version info
        st.info(f"📊 Selected: **{export_color_count} colors** - MSE: {export_metrics['mse']:.2f}, PSNR: {export_metrics['psnr']:.2f} dB, SSIM: {export_metrics['ssim']:.3f}")
        
        # Create comprehensive report with images
        with st.spinner("📄 Generating reports..."):
            report = f"""Color Quantization Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Original Image: {image.size[0]}x{image.size[1]}

COMPARISON SUMMARY:
"""
        
        # Add comparison table
        report += "Color Count | MSE | PSNR (dB) | SSIM | Compression Ratio\n"
        report += "-" * 70 + "\n"
        
        for i, (num_colors, metrics) in enumerate(zip(color_counts, metrics_list)):
            report += f"{num_colors:10d} | {metrics['mse']:5.2f} | {metrics['psnr']:8.2f} | {metrics['ssim']:4.3f} | {metrics['compression_ratio']:6.1f}x\n"
        
        report += f"""

SELECTED VERSION DETAILS:
Number of Colors: {export_color_count}

Quality Metrics:
- Mean Squared Error (MSE): {export_metrics['mse']:.2f}
- Peak Signal-to-Noise Ratio (PSNR): {export_metrics['psnr']:.2f} dB
- Structural Similarity Index (SSIM): {export_metrics['ssim']:.3f}
- Compression Ratio: {export_metrics['compression_ratio']:.1f}x

Color Palette ({export_color_count} colors):
"""
        # Generate hex colors for the export palette
        export_hex_colors = []
        for color in export_palette:
            hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            export_hex_colors.append(hex_color)
        
        for i, hex_color in enumerate(export_hex_colors):
            report += f"- Color {i+1}: {hex_color}\n"
        
        # Create visual comparison image
        def create_comparison_image(original, quantized_images, color_counts):
            """Create a side-by-side comparison image"""
            # Calculate layout
            num_images = len(quantized_images) + 1  # +1 for original
            cols = 7
            rows = 1
            
            # Calculate image dimensions
            img_width, img_height = original.size
            cell_width = img_width
            cell_height = img_height + 30  # Extra space for labels
            
            # Create canvas
            canvas_width = cols * cell_width
            canvas_height = rows * cell_height
            canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
            
            # Add original image
            canvas.paste(original, (0, 30))
            
            # Add quantized images
            for i, (quantized_img, num_colors) in enumerate(zip(quantized_images, color_counts)):
                x = (i + 1) * cell_width
                canvas.paste(quantized_img, (x, 30))
            
            # Add labels
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(canvas)
            
            # Try to use a default font, fallback to basic if not available
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            # Add labels
            labels = ["Original"] + [f"{count} colors" for count in color_counts]
            for i, label in enumerate(labels):
                x = i * cell_width + 10
                y = 5
                draw.text((x, y), label, fill='black', font=font)
            
            return canvas
        
        # Generate comparison image
        comparison_img = create_comparison_image(image, quantized_images, color_counts)
        
        # Save comparison image to buffer
        comparison_buffer = io.BytesIO()
        comparison_img.save(comparison_buffer, format='PNG')
        
        # Create HTML report with embedded image
        import base64
        
        # Convert comparison image to base64
        comparison_buffer.seek(0)
        img_base64 = base64.b64encode(comparison_buffer.getvalue()).decode()
        
        # Create HTML report
        html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Color Quantization Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .metrics-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        .metrics-table th {{ background-color: #f2f2f2; }}
        .comparison-image {{ max-width: 100%; height: auto; margin: 20px 0; }}
        .palette-section {{ margin: 20px 0; }}
        .color-swatches {{ 
            display: flex; 
            flex-wrap: wrap; 
            gap: 8px; 
            margin: 15px 0; 
            padding: 15px; 
            background-color: #f8f9fa; 
            border-radius: 8px; 
            border: 1px solid #e9ecef;
        }}
        .color-swatch {{ 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            width: 80px; 
            height: 40px; 
            border-radius: 6px; 
            border: 2px solid #ddd; 
            font-family: 'Courier New', monospace; 
            font-size: 11px; 
            font-weight: bold; 
            cursor: pointer; 
            transition: all 0.2s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .color-swatch:hover {{ 
            transform: translateY(-2px); 
            box-shadow: 0 4px 8px rgba(0,0,0,0.15); 
            border-color: #999;
        }}
        .hex-code {{ 
            text-shadow: 0 1px 2px rgba(0,0,0,0.3); 
            font-weight: bold;
        }}
        .hex-list {{ 
            background-color: #f8f9fa; 
            padding: 15px; 
            border-radius: 6px; 
            border-left: 4px solid #007bff;
        }}
        .hex-list li {{ 
            margin: 8px 0; 
            font-family: 'Courier New', monospace;
        }}
        .hex-list code {{ 
            background-color: #e9ecef; 
            padding: 2px 6px; 
            border-radius: 3px; 
            font-weight: bold;
        }}
        .analysis {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Color Quantization Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Original Image:</strong> {image.size[0]}x{image.size[1]} pixels</p>
    </div>

    <h2>Comparison Summary</h2>
    <table class="metrics-table">
        <tr>
            <th>Color Count</th>
            <th>MSE</th>
            <th>PSNR (dB)</th>
            <th>SSIM</th>
            <th>Compression Ratio</th>
        </tr>
"""
        
        # Add metrics rows
        for i, (num_colors, metrics) in enumerate(zip(color_counts, metrics_list)):
            html_report += f"""
        <tr>
            <td>{num_colors}</td>
            <td>{metrics['mse']:.2f}</td>
            <td>{metrics['psnr']:.2f}</td>
            <td>{metrics['ssim']:.3f}</td>
            <td>{metrics['compression_ratio']:.1f}x</td>
        </tr>"""
        
        html_report += f"""
    </table>

    <h2>Selected Version Details</h2>
    <p><strong>Number of Colors:</strong> {export_color_count}</p>
    
    <h3>Quality Metrics:</h3>
    <ul>
        <li><strong>Mean Squared Error (MSE):</strong> {export_metrics['mse']:.2f}</li>
        <li><strong>Peak Signal-to-Noise Ratio (PSNR):</strong> {export_metrics['psnr']:.2f} dB</li>
        <li><strong>Structural Similarity Index (SSIM):</strong> {export_metrics['ssim']:.3f}</li>
        <li><strong>Compression Ratio:</strong> {export_metrics['compression_ratio']:.1f}x</li>
    </ul>

    <h3>Color Palette ({export_color_count} colors):</h3>
    <div class="palette-section">
        <div class="color-swatches">
"""
        
        # Add color swatches with hex codes
        for i, hex_color in enumerate(export_hex_colors):
            # Determine text color based on background brightness
            r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
            brightness = (r * 299 + g * 587 + b * 114) / 1000
            text_color = '#000000' if brightness > 128 else '#ffffff'
            
            html_report += f"""
            <div class="color-swatch" style="background-color: {hex_color}; color: {text_color};" title="{hex_color}">
                <span class="hex-code">{hex_color}</span>
            </div>"""
        
        html_report += f"""
        </div>
    </div>
    <p><strong>Hex Colors:</strong></p>
    <ul class="hex-list">
"""
        
        for i, hex_color in enumerate(export_hex_colors):
            html_report += f'        <li><code>{hex_color}</code> - Color {i+1}</li>'
        
        html_report += f"""
    </ul>

    <h2>Visual Comparison</h2>
    <p>The comparison image below shows the original image alongside quantized versions with 2, 4, 8, 16, 32, and 64 colors.</p>
    <img src="data:image/png;base64,{img_base64}" alt="Color Quantization Comparison" class="comparison-image">

    <div class="analysis">
        <h3>Analysis</h3>
        <ul>
            <li><strong>2 colors:</strong> Maximum compression, significant quality loss</li>
            <li><strong>4 colors:</strong> High compression, noticeable artifacts</li>
            <li><strong>8 colors:</strong> Good balance for simple images</li>
            <li><strong>16 colors:</strong> Better quality, suitable for most images</li>
            <li><strong>32 colors:</strong> High quality, minimal artifacts</li>
            <li><strong>64 colors:</strong> Near-original quality, moderate compression</li>
        </ul>
        
        <h3>Recommendation</h3>
        <p>Based on the metrics above, consider using <strong>{export_color_count} colors</strong> for this image.</p>
    </div>
</body>
</html>
"""
        
        # Create download buttons in a grid
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download quantized image
            img_buffer = io.BytesIO()
            export_quantized.save(img_buffer, format='PNG')
            st.download_button(
                label="🖼️ Quantized Image",
                data=img_buffer.getvalue(),
                file_name=f"quantized_{export_color_count}_colors.png",
                mime="image/png",
                help="Download the quantized image"
            )
        
        with col2:
            # Download palette as image
            palette_img = Image.fromarray(palette_viz)
            palette_buffer = io.BytesIO()
            palette_img.save(palette_buffer, format='PNG')
            st.download_button(
                label="🎨 Palette Image",
                data=palette_buffer.getvalue(),
                file_name=f"palette_{export_color_count}_colors.png",
                mime="image/png",
                help="Download the color palette as an image"
            )
        
        with col3:
            # Download text report
            st.download_button(
                label="📄 Text Report",
                data=report,
                file_name=f"quantization_report_{export_color_count}_colors.txt",
                mime="text/plain",
                help="Download a simple text report with metrics and analysis"
            )
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            # Download HTML report
            st.download_button(
                label="🌐 HTML Report",
                data=html_report,
                file_name=f"quantization_report_{export_color_count}_colors.html",
                mime="text/html",
                help="Download a complete HTML report with embedded comparison image"
            )
        
        with col5:
            # Download comparison image
            st.download_button(
                label="📊 Comparison Image",
                data=comparison_buffer.getvalue(),
                file_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                help="Download the side-by-side comparison image"
            )
        
        # Create comprehensive report with images
        report = f"""Color Quantization Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Original Image: {image.size[0]}x{image.size[1]}

COMPARISON SUMMARY:
"""
        
        # Add comparison table
        report += "Color Count | MSE | PSNR (dB) | SSIM | Compression Ratio\n"
        report += "-" * 70 + "\n"
        
        for i, (num_colors, metrics) in enumerate(zip(color_counts, metrics_list)):
            report += f"{num_colors:10d} | {metrics['mse']:5.2f} | {metrics['psnr']:8.2f} | {metrics['ssim']:4.3f} | {metrics['compression_ratio']:6.1f}x\n"
        
        report += f"""

SELECTED VERSION DETAILS:
Number of Colors: {export_color_count}

Quality Metrics:
- Mean Squared Error (MSE): {export_metrics['mse']:.2f}
- Peak Signal-to-Noise Ratio (PSNR): {export_metrics['psnr']:.2f} dB
- Structural Similarity Index (SSIM): {export_metrics['ssim']:.3f}
- Compression Ratio: {export_metrics['compression_ratio']:.1f}x

Color Palette ({export_color_count} colors):
"""
        # Generate hex colors for the export palette
        export_hex_colors = []
        for color in export_palette:
            hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            export_hex_colors.append(hex_color)
        
        for i, hex_color in enumerate(export_hex_colors):
            report += f"- Color {i+1}: {hex_color}\n"
        
        # Create visual comparison image
        def create_comparison_image(original, quantized_images, color_counts):
            """Create a side-by-side comparison image"""
            # Calculate layout
            num_images = len(quantized_images) + 1  # +1 for original
            cols = 7
            rows = 1
            
            # Calculate image dimensions
            img_width, img_height = original.size
            cell_width = img_width
            cell_height = img_height + 30  # Extra space for labels
            
            # Create canvas
            canvas_width = cols * cell_width
            canvas_height = rows * cell_height
            canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
            
            # Add original image
            canvas.paste(original, (0, 30))
            
            # Add quantized images
            for i, (quantized_img, num_colors) in enumerate(zip(quantized_images, color_counts)):
                x = (i + 1) * cell_width
                canvas.paste(quantized_img, (x, 30))
            
            # Add labels
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(canvas)
            
            # Try to use a default font, fallback to basic if not available
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            # Add labels
            labels = ["Original"] + [f"{count} colors" for count in color_counts]
            for i, label in enumerate(labels):
                x = i * cell_width + 10
                y = 5
                draw.text((x, y), label, fill='black', font=font)
            
            return canvas
        
        # Generate comparison image
        comparison_img = create_comparison_image(image, quantized_images, color_counts)
        
        # Save comparison image to buffer
        comparison_buffer = io.BytesIO()
        comparison_img.save(comparison_buffer, format='PNG')
        
        # Add image info to report
        report += f"""

VISUAL COMPARISON:
The comparison image below shows the original image alongside quantized versions with 2, 4, 8, 16, 32, and 64 colors.

[COMPARISON_IMAGE_PLACEHOLDER]

ANALYSIS:
- 2 colors: Maximum compression, significant quality loss
- 4 colors: High compression, noticeable artifacts
- 8 colors: Good balance for simple images
- 16 colors: Better quality, suitable for most images
- 32 colors: High quality, minimal artifacts
- 64 colors: Near-original quality, moderate compression

RECOMMENDATION:
Based on the metrics above, consider using {export_color_count} colors for this image.
"""
        
        # Create HTML report with embedded image
        import base64
        
        # Convert comparison image to base64
        comparison_buffer.seek(0)
        img_base64 = base64.b64encode(comparison_buffer.getvalue()).decode()
        
        # Create HTML report
        html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Color Quantization Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .metrics-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        .metrics-table th {{ background-color: #f2f2f2; }}
        .comparison-image {{ max-width: 100%; height: auto; margin: 20px 0; }}
        .palette-section {{ margin: 20px 0; }}
        .color-swatches {{ 
            display: flex; 
            flex-wrap: wrap; 
            gap: 8px; 
            margin: 15px 0; 
            padding: 15px; 
            background-color: #f8f9fa; 
            border-radius: 8px; 
            border: 1px solid #e9ecef;
        }}
        .color-swatch {{ 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            width: 80px; 
            height: 40px; 
            border-radius: 6px; 
            border: 2px solid #ddd; 
            font-family: 'Courier New', monospace; 
            font-size: 11px; 
            font-weight: bold; 
            cursor: pointer; 
            transition: all 0.2s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .color-swatch:hover {{ 
            transform: translateY(-2px); 
            box-shadow: 0 4px 8px rgba(0,0,0,0.15); 
            border-color: #999;
        }}
        .hex-code {{ 
            text-shadow: 0 1px 2px rgba(0,0,0,0.3); 
            font-weight: bold;
        }}
        .hex-list {{ 
            background-color: #f8f9fa; 
            padding: 15px; 
            border-radius: 6px; 
            border-left: 4px solid #007bff;
        }}
        .hex-list li {{ 
            margin: 8px 0; 
            font-family: 'Courier New', monospace;
        }}
        .hex-list code {{ 
            background-color: #e9ecef; 
            padding: 2px 6px; 
            border-radius: 3px; 
            font-weight: bold;
        }}
        .analysis {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Color Quantization Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Original Image:</strong> {image.size[0]}x{image.size[1]} pixels</p>
    </div>

    <h2>Comparison Summary</h2>
    <table class="metrics-table">
        <tr>
            <th>Color Count</th>
            <th>MSE</th>
            <th>PSNR (dB)</th>
            <th>SSIM</th>
            <th>Compression Ratio</th>
        </tr>
"""
        
        # Add metrics rows
        for i, (num_colors, metrics) in enumerate(zip(color_counts, metrics_list)):
            html_report += f"""
        <tr>
            <td>{num_colors}</td>
            <td>{metrics['mse']:.2f}</td>
            <td>{metrics['psnr']:.2f}</td>
            <td>{metrics['ssim']:.3f}</td>
            <td>{metrics['compression_ratio']:.1f}x</td>
        </tr>"""
        
        html_report += f"""
    </table>

    <h2>Selected Version Details</h2>
    <p><strong>Number of Colors:</strong> {export_color_count}</p>
    
    <h3>Quality Metrics:</h3>
    <ul>
        <li><strong>Mean Squared Error (MSE):</strong> {export_metrics['mse']:.2f}</li>
        <li><strong>Peak Signal-to-Noise Ratio (PSNR):</strong> {export_metrics['psnr']:.2f} dB</li>
        <li><strong>Structural Similarity Index (SSIM):</strong> {export_metrics['ssim']:.3f}</li>
        <li><strong>Compression Ratio:</strong> {export_metrics['compression_ratio']:.1f}x</li>
    </ul>

    <h3>Color Palette ({export_color_count} colors):</h3>
    <div class="palette-section">
        <div class="color-swatches">
"""
        
        # Add color swatches with hex codes
        for i, hex_color in enumerate(export_hex_colors):
            # Determine text color based on background brightness
            r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
            brightness = (r * 299 + g * 587 + b * 114) / 1000
            text_color = '#000000' if brightness > 128 else '#ffffff'
            
            html_report += f"""
            <div class="color-swatch" style="background-color: {hex_color}; color: {text_color};" title="{hex_color}">
                <span class="hex-code">{hex_color}</span>
            </div>"""
        
        html_report += f"""
        </div>
    </div>
    <p><strong>Hex Colors:</strong></p>
    <ul class="hex-list">
"""
        
        for i, hex_color in enumerate(export_hex_colors):
            html_report += f'        <li><code>{hex_color}</code> - Color {i+1}</li>'
        
        html_report += f"""
    </ul>

    <h2>Visual Comparison</h2>
    <p>The comparison image below shows the original image alongside quantized versions with 2, 4, 8, 16, 32, and 64 colors.</p>
    <img src="data:image/png;base64,{img_base64}" alt="Color Quantization Comparison" class="comparison-image">

    <div class="analysis">
        <h3>Analysis</h3>
        <ul>
            <li><strong>2 colors:</strong> Maximum compression, significant quality loss</li>
            <li><strong>4 colors:</strong> High compression, noticeable artifacts</li>
            <li><strong>8 colors:</strong> Good balance for simple images</li>
            <li><strong>16 colors:</strong> Better quality, suitable for most images</li>
            <li><strong>32 colors:</strong> High quality, minimal artifacts</li>
            <li><strong>64 colors:</strong> Near-original quality, moderate compression</li>
        </ul>
        
        <h3>Recommendation</h3>
        <p>Based on the metrics above, consider using <strong>{export_color_count} colors</strong> for this image.</p>
    </div>
</body>
</html>
"""
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.download_button(
                label="Download Text Report",
                data=report,
                file_name=f"quantization_report_{export_color_count}_colors.txt",
                mime="text/plain"
            )
        
        with col5:
            st.download_button(
                label="Download HTML Report",
                data=html_report,
                file_name=f"quantization_report_{export_color_count}_colors.html",
                mime="text/html"
            )
        
        with col6:
            st.download_button(
                label="Download Comparison Image",
                data=comparison_buffer.getvalue(),
                file_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )
    
    # History information
    if st.session_state.history:
        st.sidebar.write(f"History: {len(st.session_state.history)} states")
        st.sidebar.write(f"Current: {st.session_state.current_index + 1}")
