import cv2
import numpy as np
from pathlib import Path

# step 1
def load_expression_image(image_path: Path) -> np.ndarray:
    """
    Load a handwritten expression image as grayscale.

    Takes a path to an image file and returns a grayscale numpy array (H, W).
    Raises FileNotFoundError if the image doesn't exist or can't be read.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not decode image: {image_path}")
    
    return img

# step 2
def threshold_expression(img: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale image to binary.

    Symbols become white (255), background becomes black (0).
    Uses Otsu's method so the threshold adapts to each image automatically.
    """

    _, binary = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary

# step 3
def find_symbol_boxes(binary: np.ndarray) -> list[tuple]:
    """
    Find contours in a binary image and return their bounding boxes.

    Takes a binary image (white symbols on black background).
    Returns list of (x, y, w, h) tuples, one per detected contour.
    Unfiltered and unsorted.
    """

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w, h))

    return boxes

# step 4
def filter_boxes(
        boxes: list[tuple],
        min_w: int = 5,
        min_h: int = 5,
        min_area: int = 150,
        ) -> list[tuple]:
    """
    Remove noise contours that are too small to be real symbols.

    Filters on three things independently:
    - min_w: minimum width in pixels (catches thin vertical artifacts)
    - min_h: minimum height in pixels (catches thin horizontal artifacts)
    - min_area: minimum bounding-box area in px² (catches small blobs)
    """
    filtered = []
    for (x, y, w, h) in boxes:
        if w < min_w or h < min_h:
            continue
        if w * h < min_area:
            continue
        filtered.append((x, y, w, h))
 
    return filtered

# step 5
def sort_boxes_left_to_right(boxes: list[tuple]) -> list[tuple]:
    """
    Sort bounding boxes in reading order (left to right by x coordinate).

    Index 0 is the leftmost symbol.
    """
    return sorted(boxes, key=lambda b: b[0])


# step 6
def crop_symbols(
    binary: np.ndarray,
    boxes: list[tuple],
    pad: int = 4,
) -> list[np.ndarray]:
    """
    Crop each bounding box out of the binary image.

    A small padding is added so strokes at the very edge are not clipped.
    Padding is clamped so it never exceeds the image boundary.
    """

    H, W = binary.shape
    crops = []
    for (x, y, w, h) in boxes:
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)
        crop = binary[y1:y2, x1:x2]
        crops.append(crop)
 
    return crops


# step 7 - master pipeline

def segment_expression(
        image_path: Path,
        min_w: int = 5,
        min_h: int = 5,
        min_area: int = 150,
        pad: int = 4,
) -> tuple[list[tuple], list[np.ndarray], np.ndarray]:
    """
    Full segmentation pipeline: load -> threshold -> contours -> filter -> sort -> crop.

    Returns (boxes, crops, binary) where:
    - boxes: list of (x, y, w, h) tuples, filtered and sorted left-to-right
    - crops: list of numpy arrays, one cropped symbol per box
    - binary: the thresholded binary image (useful for debugging)
    """
    img    = load_expression_image(image_path)
    binary = threshold_expression(img)
    boxes  = find_symbol_boxes(binary)
    boxes  = filter_boxes(boxes, min_w=min_w, min_h=min_h, min_area=min_area)
    boxes  = sort_boxes_left_to_right(boxes)
    crops  = crop_symbols(binary, boxes, pad=pad)
 
    return boxes, crops, binary