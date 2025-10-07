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
   git clone <repo-url>
   cd graph-extractor
