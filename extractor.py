import cv2
import numpy as np
import pytesseract
import re
import csv
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.interpolate import interp1d

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    
    # Convert to grayscale and apply Gaussian blur for better edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return img, gray, thresh

def detect_axes(thresh, img):
    edges = cv2.Canny(thresh, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=5)
    
    x_axis, y_axis = None, None
    max_x_length = max_y_length = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)

        # Check for horizontal line near bottom (X-axis)
        if abs(y1 - y2) < 15 and y1 > img.shape[0] * 0.8 and length > max_x_length:
            x_axis = (x1, y1, x2, y2)
            max_x_length = length

        # Check for vertical line near left (Y-axis)
        if abs(x1 - x2) < 15 and x1 < img.shape[1] * 0.2 and length > max_y_length:
            y_axis = (x1, y1, x2, y2)
            max_y_length = length


    return x_axis, y_axis

def crop_graph(img, x_axis, y_axis):
    x1, y1, x2, y2 = x_axis
    x3, y3, x4, y4 = y_axis
    
    cropped = img[y4:y3, x3:x2]
    return cropped

def extract_axis_values(img, x_axis, y_axis):
    h, w = img.shape[:2]
    
    x1, y1, x2, y2 = x_axis
    x_text_region = img[y1:y1+30, x1:x2]
    cv2.rectangle(img, (x1, y1), (x2, y1+30), (0, 255, 0), 2)
    x_text_region_gray = cv2.cvtColor(x_text_region, cv2.COLOR_BGR2GRAY)
    _, x_text_region = cv2.threshold(x_text_region_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    x_text = pytesseract.image_to_string(x_text_region, config='--psm 6 --oem 3')
    
    x3, y3, x4, y4 = y_axis
    y_text_region = img[y4:y3, x3-45:x3]
    cv2.rectangle(img, (x3-45, y4), (x3, y3), (255, 0, 0), 2)
    y_text_region_gray = cv2.cvtColor(y_text_region, cv2.COLOR_BGR2GRAY)
    _, y_text_region = cv2.threshold(y_text_region_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    y_text = pytesseract.image_to_string(y_text_region, config='--psm 6 --oem 3')

    cv2.imshow("Values extraction" , img)

    y_values = [float(num) for num in re.findall(r'\d+\.\d+', y_text)]
    x_values = [int(num) for num in re.findall(r'\d+', x_text)]

    # print("x_val: ", x_values)
    # print("y_val: ", y_values)
    
    return x_values, y_values


    img_copy = img.copy()
    h, w = img.shape[:2]

    # Step 1: Detect edges and grid lines using Hough Transform
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    cv2.imshow("check2", edges)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50)
    if lines is None:
        print("No lines detected")
        return

    # Separate vertical and horizontal lines
    vertical_lines = []
    horizontal_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) < 10:
            vertical_lines.append(x1)
        elif abs(y1 - y2) < 10:
            horizontal_lines.append(y1)

    # Remove outermost lines by trimming ends
    vertical_lines = sorted(list(set(vertical_lines)))
    horizontal_lines = sorted(list(set(horizontal_lines)))

    # Remove outermost (rectangle) lines
    vertical_lines = vertical_lines[1:-1]
    horizontal_lines = horizontal_lines[1:-1]

    # Match detected grid lines to x_vals and y_vals
    if len(vertical_lines) != len(x_vals):
        print(f"Expected {len(x_vals)} vertical lines, got {len(vertical_lines)}")
    if len(horizontal_lines) != len(y_vals):
        print(f"Expected {len(y_vals)} horizontal lines, got {len(horizontal_lines)}")

    # Draw vertical lines and x values
    for x, val in zip(vertical_lines, x_vals):
        cv2.line(img_copy, (x, 0), (x, h), (0, 255, 0), 1)
        cv2.putText(img_copy, str(val), (x - 10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw horizontal lines and y values
    for y, val in zip(horizontal_lines, y_vals):
        cv2.line(img_copy, (0, y), (w, y), (255, 0, 0), 1)
        cv2.putText(img_copy, str(val), (10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Show the result
    cv2.imshow("Detected Grid with Values", img_copy)

def cluster_positions(positions, threshold=10):
    positions.sort()
    clustered = []
    group = [positions[0]]

    for pos in positions[1:]:
        if abs(pos - group[-1]) <= threshold:
            group.append(pos)
        else:
            clustered.append(int(np.mean(group)))
            group = [pos]
    clustered.append(int(np.mean(group)))
    return clustered

def detect_grid_lines(img, x_vals, y_vals):
    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    h, w = binary.shape

    # Detect vertical lines by scanning columns
    vertical_lines = []
    for x in range(w):
        if x < 2 or x > w - 2:
            continue 
        col = binary[:, x]
        if np.count_nonzero(col) > h * 0.8:
            vertical_lines.append(x)

    # Detect horizontal lines by scanning rows
    horizontal_lines = []
    for y in range(h):
        if y < 4 or y > h - 4:
            continue 
        row = binary[y, :]
        if np.count_nonzero(row) > w * 0.8:
            horizontal_lines.append(y)

    # Cluster close lines
    vertical_lines = cluster_positions(vertical_lines)
    horizontal_lines = cluster_positions(horizontal_lines)

    # print(horizontal_lines)
    # print(vertical_lines)

    x_vals = fill_linear(x_vals, len(vertical_lines))
    y_vals = fill_linear(y_vals, len(horizontal_lines))

    print("X-axis:", x_vals)
    print("Y-axis:", y_vals)
    
    # Draw vertical lines
    for x, val in zip(vertical_lines, x_vals):
        cv2.line(img_copy, (x, 0), (x, h), (0, 255, 0), 1)
        cv2.putText(img_copy, str(val), (x - 10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw horizontal lines
    for y, val in zip(horizontal_lines, y_vals):
        cv2.line(img_copy, (0, y), (w, y), (255, 0, 0), 1)
        cv2.putText(img_copy, str(val), (10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow("Grid lines", img_copy)
    return vertical_lines, horizontal_lines

def fill_linear(seq, final_len):
    if len(seq) < 2 or final_len <= 1:
        return seq[:final_len]
    
    start = seq[0]
    end = seq[-1]
    step = (end - start) / (final_len - 1)

    return [round(start + i * step, 10) for i in range(final_len)]

def digitize_graph_data(img, x_pixel_positions, y_pixel_positions, x_values, y_values, step=2):

    x_values = fill_linear(x_values, len(x_pixel_positions))
    y_values = fill_linear(y_values, len(y_pixel_positions))

    # Convert image to grayscale and binary
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Get non-zero (white) pixel coordinates
    points = cv2.findNonZero(binary)

    points = points.reshape(-1, 2)
    points = points[points[:, 0].argsort()] 

    # Set up interpolation functions
    x_interp = interp1d(x_pixel_positions, x_values, fill_value='extrapolate')
    y_interp = interp1d(y_pixel_positions, y_values, fill_value='extrapolate')

    def pixel_to_x(px):
        return float(x_interp(px))

    def pixel_to_y(py):
        return float(y_interp(py))

    h, w = img.shape[:2]
    boundary_margin = 10  

    data_points = []
    for i in range(0, len(points), step):
        px, py = points[i]
        if (boundary_margin < px < w - boundary_margin and
            boundary_margin < py < h - boundary_margin):
            x_val = pixel_to_x(px)
            y_val = pixel_to_y(py)
            data_points.append((x_val, y_val))
            cv2.circle(img, (px, py), 2, (0, 0, 255), -1)

    cv2.imshow("Data points", img)
    y_val = [y for _, y in data_points]

    return data_points, y_val

def analyze_graph(y):

    stats = {
        'Mean': np.mean(y),
        'Min': np.min(y),
        'Max': np.max(y),
        'RMS': np.sqrt(np.mean(np.square(y))),
        'Peak-to-Valley': np.max(y) - np.min(y),
        'Skewness': skew(y),
        'Kurtosis': kurtosis(y)  
    }

    for key, value in stats.items():
        print(f"{key}: {value:.4f}")

    return stats

def make_csv(datapoints, csv_path):
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['X', 'Y'])
        writer.writerows(datapoints)

img, gray, thresh = preprocess_image('graph/graph3.png')
x_axis, y_axis = detect_axes(thresh, img)
x_values, y_values = extract_axis_values(img, x_axis, y_axis)
cropped_graph = crop_graph(img, x_axis, y_axis)
x_pos, y_pos = detect_grid_lines(cropped_graph, x_values, y_values)
data_points, y_val = digitize_graph_data(cropped_graph, x_pos, y_pos, x_values, y_values)
analyze_graph(y_val)
make_csv(data_points, 'csv/graph3.csv')

cv2.waitKey(0)
cv2.destroyAllWindows()