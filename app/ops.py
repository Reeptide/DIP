"""Image processing operations and the operation catalog.

The original comments said "12 operations" but there are 13 entries below -
just an off-by-one in the old comments, corrected here.

OPERATION_MAP is hoisted to module level: the original worker rebuilt this
dict from scratch on every single tile inside process_tile(). Harmless
functionally, wasteful at scale - a no-op fix bundled into the move.
"""
import cv2
import numpy as np

# Display names shown in the upload UI, and the source of truth for which
# operation keys are valid.
PROCESSING_OPERATIONS = {
    'grayscale': 'Grayscale Conversion',
    'color_inversion': 'Color Inversion',
    'blur_gaussian': 'Gaussian Blur',
    'blur_median': 'Median Blur',
    'blur_bilateral': 'Bilateral Blur (Edge-Preserving)',
    'edge_canny': 'Canny Edge Detection',
    'edge_sobel': 'Sobel Edge Detection',
    'edge_laplacian': 'Laplacian Edge Detection',
    'sharpen': 'Image Sharpening',
    'histogram_equalization': 'Histogram Equalization',
    'threshold_binary': 'Binary Thresholding',
    'contour_detection': 'Contour Detection',
    'morphological_opening': 'Morphological Opening',
}


class ImageProcessor:
    """Image processing operations using OpenCV. All static: no per-call
    instantiation needed."""

    @staticmethod
    def grayscale(tile_image):
        gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def color_inversion(tile_image):
        return cv2.bitwise_not(tile_image)

    @staticmethod
    def blur_gaussian(tile_image, kernel_size=51):
        return cv2.GaussianBlur(tile_image, (kernel_size, kernel_size), 0)

    @staticmethod
    def blur_median(tile_image, kernel_size=15):
        return cv2.medianBlur(tile_image, kernel_size)

    @staticmethod
    def blur_bilateral(tile_image, diameter=9, sigma_color=75, sigma_space=75):
        return cv2.bilateralFilter(tile_image, diameter, sigma_color, sigma_space)

    @staticmethod
    def edge_canny(tile_image, low_threshold=50, high_threshold=150):
        gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def edge_sobel(tile_image):
        gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
        sobel = np.uint8(255 * sobel / np.max(sobel))
        return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def edge_laplacian(tile_image):
        gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        return cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def sharpen(tile_image):
        kernel = np.array([[-1, -1, -1],
                            [-1, 9, -1],
                            [-1, -1, -1]])
        return cv2.filter2D(tile_image, -1, kernel)

    @staticmethod
    def histogram_equalization(tile_image):
        if len(tile_image.shape) == 3:
            ycrcb = cv2.cvtColor(tile_image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return cv2.equalizeHist(tile_image)

    @staticmethod
    def threshold_binary(tile_image, threshold_value=127):
        gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def contour_detection(tile_image):
        gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = tile_image.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        return result

    @staticmethod
    def morphological_opening(tile_image, kernel_size=5):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(tile_image, cv2.MORPH_OPEN, kernel)


OPERATION_MAP = {
    'grayscale': ImageProcessor.grayscale,
    'color_inversion': ImageProcessor.color_inversion,
    'blur_gaussian': ImageProcessor.blur_gaussian,
    'blur_median': ImageProcessor.blur_median,
    'blur_bilateral': ImageProcessor.blur_bilateral,
    'edge_canny': ImageProcessor.edge_canny,
    'edge_sobel': ImageProcessor.edge_sobel,
    'edge_laplacian': ImageProcessor.edge_laplacian,
    'sharpen': ImageProcessor.sharpen,
    'histogram_equalization': ImageProcessor.histogram_equalization,
    'threshold_binary': ImageProcessor.threshold_binary,
    'contour_detection': ImageProcessor.contour_detection,
    'morphological_opening': ImageProcessor.morphological_opening,
}
