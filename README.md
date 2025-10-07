# Graph Extractor

Automatically extracts data points from graph images using **computer vision**, **OCR**, and **signal processing**. The extracted data can be analyzed and exported to CSV for further use.

## Features

- Detect X and Y axes from graphs
- Extract axis values using OCR (Tesseract)
- Detect grid lines and map pixel positions to actual values
- Digitize graph data points
- Highlight deviation points (local maxima/minima)
- Compute statistics (mean, min, max, RMS, skewness, kurtosis)
- Export data to CSV for analysis

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/meekhumor/Graph-Extractor.git
   cd Graph-Extractor
   ```
2. Install dependencies:
   ```bash
   pip install opencv-python numpy pytesseract scipy pandas
   ```
3. Make sure [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) is installed and added to your PATH.

## Usage

1. Place your graph image in the `graph/` directory (e.g., `graph/your_graph.png`). The script expects a simple line graph with grid lines for best results.
2. Run the main script:
   ```bash
   python extractor.py
   ```
   - By default, it processes `graph/graph3.png`.
   - To use a different image, modify the line:
     ```python
     preprocess_image('graph/your_image.png')
     ```
   - OpenCV windows will display visualizations of axes detection, grid lines, data points, and deviations.
   - Extracted data is saved as `csv/graph3.csv` (columns: X, Y).
   - Statistics (mean, min, max, RMS, peak-to-valley, skewness, kurtosis) are printed to the console.
3. Press any key in the OpenCV windows to close them.

## Example Output

**Console Stats:**
```
Mean: 5.2341
Min: 1.0000
Max: 10.0000
RMS: 5.6789
Peak-to-Valley: 9.0000
Skewness: 0.1234
Kurtosis: -0.5678
```

**CSV File (`csv/graph3.csv`):**
```
X,Y
0.0,4.5
1.0,5.2
2.0,3.8
...
```

### Screenshots

**Value Extraction:**  
![Value Extraction](https://github.com/user-attachments/assets/d7df1a38-ca7f-4238-83cc-9a4350a7b1e3)

**Grid Lines:**  
![Grid Lines](https://github.com/user-attachments/assets/03a216b0-de03-4363-9aea-2086e688a158)

**Data Points:**  
![Data Points](https://github.com/user-attachments/assets/2afe7b5b-cfae-4465-bac4-4993803b20ff)

## How It Works

1. **Preprocessing:** Loads the image, converts to grayscale, and applies thresholding for edge detection.
2. **Axis Detection:** Uses Hough Line Transform to identify horizontal (X-axis) and vertical (Y-axis) lines.
3. **Value Extraction:** Crops text regions from axes and uses Tesseract OCR to parse numerical values.
4. **Grid Detection:** Scans for vertical/horizontal lines, clusters them, and maps to interpolated axis values.
5. **Digitization:** Finds non-zero pixels (data points), interpolates pixel coordinates to real values.
6. **Analysis:** Detects local extrema, computes statistics, and exports to CSV.


## Directory Structure

```
meekhumor-graph-extractor/
├── README.md
├── extractor.py
├── graph/          # Input graph images (e.g., graph3.png)
└── csv/            # Output CSV files (auto-created)
```

## Limitations

- Best for simple 2D line plots with visible grid lines.
- OCR accuracy depends on text clarity; may fail on handwritten or stylized labels.
- No support for logarithmic scales or multi-line graphs (extendable).
- Requires manual image path updates; consider adding command-line arguments for production use.

## Contributing

Contributions are welcome! Fork the repo, make changes, and submit a pull request. For major changes, please open an issue first.
