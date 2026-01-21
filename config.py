import cv2

SUFFIX_EXTRACTED_IMAGES_FOLDER = "extracted_images"
SUFFIX_EXTRACTED_TABLES_FOLDER = "extracted_tables"
SUFFIX_EXTRACTED_REFINED_TABLE_BORDERS_FOLDER = "extracted_refined_table_borders"
SUFFIX_SAVED_PAGES_IMAGE_FORMAT = ".png"
PREFIX_SAVED_PAGES_IMAGE = "page_"
ACCEPTED_IMAGE_FORMATS = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
ACCEPTED_PDF_FORMATS = ".pdf"
PAGE_EXTRACTION_ZOOM_FACTOR = 300 / 72  # Assuming A4 size at 300 DPI
PAGE_EXTRACTION_PIXMAP_ALPHA = False
PAGE_EXTRACTION_COLORSPACE = "RGB"  # Options: "RGB", "L" (grayscale)
TABLE_EXTRACTION_COLORSPACE_CONVERSION = cv2.COLOR_BGR2GRAY  # Convert to grayscale
TABLE_EXTRACTION_BINARIZATION_THRESHOLD = 128  # Grayscale threshold for binarization
TABLE_EXTRACTION_BINARIZATION_MAXVAL = 255  # Maximum value for binarization
TABLE_EXTRACTION_BINARIZATION_TYPE = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU 
TABLE_EXTRACTION_BINARIZATION_STRUCTURING_ELEMENT_SHAPE = cv2.MORPH_RECT  # Shape of the structuring element
TABLE_EXTRACTION_BINARIZATION_STRUCTURING_MORPHOLOGY_OPERATION = cv2.MORPH_OPEN  # Morphological operation https://www.geeksforgeeks.org/python/python-opencv-morphological-operations/
TABLE_EXTRACTION_BINARIZATION_STRUCTURING_MORPHOLOGY_OPERATION_ITERATIONS = 1  # Number of iterations for morphological operation
TABLE_EXTRACTION_BINARIZATION_HORIZONTAL_KERNEL_SIZE = (5, 1)  # Kernel size for horizontal line detection
TABLE_EXTRACTION_BINARIZATION_VERTICAL_KERNEL_SIZE = (1, 5)  # Kernel size for vertical line detection
TABLE_EXTRACTION_BINARIZATION_KERNEL_ARRAY_SHAPE = (7, 7)  
TABLE_EXTRACTION_BINARIZATION_DILATION_ITERATIONS = 1  # Number of dilation iterations
TABLE_BORDER_REFINEMENT_THRESHOLD = 20  # Threshold for refining table borders
TABLE_BORDER_REFINEMENT_SEARCH_WINDOW = 25  # Search window size for border refinement X and Y
TABLE_BORDER_REFINEMENT_X_OFFSET = 300  # X offset for refining table borders
TABLE_BORDER_REFINEMENT_Y_OFFSET = 300  # Y offset for refining table borders




