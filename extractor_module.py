"""
Refactored Table Segmentation Module

Description: This module contains functionalities destined to perform table segmentation from images and PDF documents.

Date: 2025-11-27
Version : 0.5.0
Author: Samy Ibriz Pelaez
Contact: helpdesk@unil.ch; samy.ibrizpelaez@unil.ch; samyibrizpelaez.dev@gmail.com
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Protocol
import logging
from contextlib import contextmanager
import cv2
import numpy as np
from PIL import Image
import pymupdf
import time
from datetime import datetime
from config import *


@dataclass(frozen=True)
class ProcessingConfig:
    """Configuration"""
    
    # Image extraction settings
    zoom_factor: float = PAGE_EXTRACTION_ZOOM_FACTOR
    color_mode: str = PAGE_EXTRACTION_COLORSPACE
    alpha_channel: bool = PAGE_EXTRACTION_PIXMAP_ALPHA
    
    # Table extraction settings
    binarization_threshold: int = TABLE_EXTRACTION_BINARIZATION_THRESHOLD
    binarization_max_value: int = TABLE_EXTRACTION_BINARIZATION_MAXVAL
    horizontal_kernel_size: tuple = TABLE_EXTRACTION_BINARIZATION_HORIZONTAL_KERNEL_SIZE
    vertical_kernel_size: tuple = TABLE_EXTRACTION_BINARIZATION_VERTICAL_KERNEL_SIZE
    dilation_kernel_size: tuple = TABLE_EXTRACTION_BINARIZATION_KERNEL_ARRAY_SHAPE
    dilation_iterations: int = TABLE_EXTRACTION_BINARIZATION_DILATION_ITERATIONS
    
    # Border refinement settings
    refinement_threshold: int = TABLE_BORDER_REFINEMENT_THRESHOLD
    search_window_size: int = TABLE_BORDER_REFINEMENT_SEARCH_WINDOW
    x_offset: int = TABLE_BORDER_REFINEMENT_X_OFFSET
    y_offset: int = TABLE_BORDER_REFINEMENT_Y_OFFSET
    
    # File extensions
    supported_image_formats: tuple = ACCEPTED_IMAGE_FORMATS
    supported_pdf_format: str = ACCEPTED_PDF_FORMATS
    output_image_format: str = SUFFIX_SAVED_PAGES_IMAGE_FORMAT


class ProcessingStage(Enum):
    """Stages"""
    INITIALIZATION = "initialization"
    PDF_LOADING = "pdf_loading"
    IMAGE_EXTRACTION = "image_extraction"
    TABLE_EXTRACTION = "table_extraction"
    BORDER_REFINEMENT = "border_refinement"
    COMPLETION = "completion"


class ExtractionMethod(Enum):
    """Extraction methods"""
    CONTOUR_DETECTION = "contour_detection"
    LINE_DETECTION = "line_detection"
    HYBRID_DETECTION = "hybrid_detection"
    

class TableProcessingError(Exception):
    """Base exception for table processing operations."""
    pass


class InvalidInputError(TableProcessingError):
    """Raised when input validation fails."""
    pass


class ProcessingStageError(TableProcessingError):
    """Raised when a processing stage fails."""
    pass

class ProgressObserver(Protocol):
    """Protocol for progress observation."""
    
    def on_stage_started(self, stage: ProcessingStage, message: str) -> None:
        """Called when a processing stage starts."""
        ...
    
    def on_stage_completed(self, stage: ProcessingStage, message: str) -> None:
        """Called when a processing stage completes."""
        ...
    
    def on_progress_update(self, current: int, total: int, message: str) -> None:
        """Called to update progress within a stage."""
        ...
    
    def on_error(self, stage: ProcessingStage, error: Exception) -> None:
        """Called when an error occurs."""
        ...
        

@dataclass(frozen=True)
class ImageMetadata:
    """Metadata for processed images."""
    source_path: Path
    output_path: Path
    width: int
    height: int
    format: str
    processing_timestamp: float = field(default_factory=lambda: __import__('time').time())


@dataclass(frozen=True)
class BoundingBox:
    """Represents a bounding box for table extraction."""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def area(self) -> int:
        """Calculate the area of the bounding box."""
        return self.width * self.height
    
    def is_valid(self) -> bool:
        """Check if the bounding box is valid."""
        return self.width > 0 and self.height > 0


@dataclass(frozen=True)
class ProcessingOptions:
    """Options for table processing operations."""
    extract_images: bool = False
    extract_tables: bool = False
    refine_borders: bool = False
    extraction_method: ExtractionMethod = ExtractionMethod.LINE_DETECTION


class ImageProcessor(ABC):
    
    @abstractmethod
    def process(self, image_path: Path, output_dir: Path, config: ProcessingConfig) -> Optional[ImageMetadata]:
        pass
    
    @abstractmethod
    def validate_input(self, image_path: Path) -> bool:
        pass


class ExtractionStrategy(ABC):
    
    @abstractmethod
    def extract(self, image_path: Path, output_dir: Path, config: ProcessingConfig) -> Optional[BoundingBox]:
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        pass


class ContourExtractionStrategy(ExtractionStrategy):
    """Strategy for contour-based table extraction."""
    
    def extract(self, image_path: Path, output_dir: Path, config: ProcessingConfig) -> Optional[BoundingBox]:
        """Extract table using contour detection."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ProcessingStageError(f"Cannot load image: {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, TABLE_EXTRACTION_COLORSPACE_CONVERSION)
            
            # Apply thresholding
            _, thresh = cv2.threshold(
                gray, 
                config.binarization_threshold, 
                config.binarization_max_value,
                TABLE_EXTRACTION_BINARIZATION_TYPE
            )
            
            # Create morphological kernel and apply dilation
            kernel = np.ones((15, 15), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Extract and save table region
            table_image = img[y:y+h, x:x+w]
            output_path = output_dir / image_path.name
            cv2.imwrite(str(output_path), table_image)
            
            return BoundingBox(x, y, w, h)
            
        except Exception as e:
            raise ProcessingStageError(f"Contour extraction failed: {e}") from e
    
    def get_method_name(self) -> str:
        return "Contour Detection"


class LineDetectionStrategy(ExtractionStrategy):
    """Strategy for line-based table extraction."""
    
    def extract(self, image_path: Path, output_dir: Path, config: ProcessingConfig) -> Optional[BoundingBox]:
        """Extract table using line detection."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ProcessingStageError(f"Cannot load image: {image_path}")
            
            # Convert to grayscale and apply threshold
            gray = cv2.cvtColor(img, TABLE_EXTRACTION_COLORSPACE_CONVERSION)
            _, thresh = cv2.threshold(
                gray,
                config.binarization_threshold,
                config.binarization_max_value,
                TABLE_EXTRACTION_BINARIZATION_TYPE
            )
            
            # Detect horizontal lines
            h_kernel = cv2.getStructuringElement(TABLE_EXTRACTION_BINARIZATION_STRUCTURING_ELEMENT_SHAPE, config.horizontal_kernel_size)
            h_lines = cv2.morphologyEx(thresh, TABLE_EXTRACTION_BINARIZATION_STRUCTURING_MORPHOLOGY_OPERATION, h_kernel, iterations=TABLE_EXTRACTION_BINARIZATION_STRUCTURING_MORPHOLOGY_OPERATION_ITERATIONS)
            
            # Detect vertical lines
            v_kernel = cv2.getStructuringElement(TABLE_EXTRACTION_BINARIZATION_STRUCTURING_ELEMENT_SHAPE, config.vertical_kernel_size)
            v_lines = cv2.morphologyEx(thresh, TABLE_EXTRACTION_BINARIZATION_STRUCTURING_MORPHOLOGY_OPERATION, v_kernel, iterations=TABLE_EXTRACTION_BINARIZATION_STRUCTURING_MORPHOLOGY_OPERATION_ITERATIONS)
            
            # Combine lines
            table_mask = cv2.add(h_lines, v_lines)
            
            # Apply dilation
            dilation_kernel = np.ones(config.dilation_kernel_size, np.uint8)
            dilated = cv2.dilate(table_mask, dilation_kernel, iterations=config.dilation_iterations)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Get largest contour and refine bounding box
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Refine bounding box using contour points
            contour_points = largest_contour.reshape(-1, 2)
            top_left = min(contour_points, key=lambda pt: pt[0] + pt[1])
            
            x_diff = x - top_left[0]
            y_diff = y - top_left[1]
            x, y = top_left
            w += x_diff
            h += y_diff
            
            # Extract and save table region
            table_image = img[y:y+h, x:x+w]
            output_path = output_dir / image_path.name
            cv2.imwrite(str(output_path), table_image)
            
            return BoundingBox(x, y, w, h)
            
        except Exception as e:
            raise ProcessingStageError(f"Line detection failed: {e}") from e
    
    def get_method_name(self) -> str:
        return "Line Detection"


class ExtractionStrategyFactory:
    """Factory for creating extraction strategies."""
    
    _strategies: Dict[ExtractionMethod, type[ExtractionStrategy]] = {
        ExtractionMethod.CONTOUR_DETECTION: ContourExtractionStrategy,
        ExtractionMethod.LINE_DETECTION: LineDetectionStrategy,
    }
    
    @classmethod
    def create_strategy(cls, method: ExtractionMethod) -> ExtractionStrategy:
        """Create an extraction strategy instance."""
        if method not in cls._strategies:
            raise ValueError(f"Unsupported extraction method: {method}")
        
        strategy_class = cls._strategies[method]
        return strategy_class()
    
    @classmethod
    def register_strategy(cls, method: ExtractionMethod, strategy_class: type[ExtractionStrategy]) -> None:
        """Register a new extraction strategy."""
        cls._strategies[method] = strategy_class
    
    @classmethod
    def get_available_methods(cls) -> List[ExtractionMethod]:
        """Get list of available extraction methods."""
        return list(cls._strategies.keys())


class ImageExtractor(ImageProcessor):
    """Processor for extracting images from PDF documents."""
    
    def process(self, pdf_path: Path, output_dir: Path, config: ProcessingConfig, observer: Optional[ProgressObserver] = None) -> List[ImageMetadata]:
        """Extract images from PDF document."""
        try:
            doc = pymupdf.open(str(pdf_path))
            images_metadata = []
            
            matrix = pymupdf.Matrix(config.zoom_factor, config.zoom_factor)
            total_pages = len(doc)
            
            if observer:
                observer.on_progress_update(
                    0, total_pages,
                    f"📄 Found {total_pages} pages in PDF"
                )
            
            for page_num in range(total_pages):
                if observer:
                    observer.on_progress_update(
                        page_num + 1, total_pages,
                        f"🖼️ Extracting page {page_num + 1}/{total_pages}"
                    )
                
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=matrix, alpha=config.alpha_channel)
                img = Image.frombytes(config.color_mode, [pix.width, pix.height], pix.samples)
                
                output_path = output_dir / f"{PREFIX_SAVED_PAGES_IMAGE}{page_num + 1}{config.output_image_format}"
                img.save(str(output_path))
                
                if observer:
                    observer.on_progress_update(
                        page_num + 1, total_pages,
                        f"✅ Saved page {page_num + 1}: {output_path.name}"
                    )
                
                metadata = ImageMetadata(
                    source_path=pdf_path,
                    output_path=output_path,
                    width=pix.width,
                    height=pix.height,
                    format=config.output_image_format
                )
                images_metadata.append(metadata)
            
            doc.close()
            return images_metadata
            
        except Exception as e:
            raise ProcessingStageError(f"Image extraction failed: {e}") from e
    
    def validate_input(self, pdf_path: Path) -> bool:
        """Validate PDF input."""
        return pdf_path.exists() and pdf_path.suffix.lower() == ACCEPTED_PDF_FORMATS


class TableExtractor(ImageProcessor):
    """Processor for extracting tables from images."""
    
    def __init__(self, strategy: ExtractionStrategy):
        self._strategy = strategy
    
    def process(self, image_path: Path, output_dir: Path, config: ProcessingConfig, observer: Optional[ProgressObserver] = None) -> Optional[ImageMetadata]:
        """Extract table from image."""
        if not self.validate_input(image_path):
            if observer:
                observer.on_progress_update(
                    1, 1,
                    f"⚠️ Skipping invalid image: {image_path.name}"
                )
            return None
        
        bounding_box = self._strategy.extract(image_path, output_dir, config)
        if bounding_box and bounding_box.is_valid():
            output_path = output_dir / image_path.name
            
            return ImageMetadata(
                source_path=image_path,
                output_path=output_path,
                width=bounding_box.width,
                height=bounding_box.height,
                format=image_path.suffix
            )
        return None
    
    def validate_input(self, image_path: Path) -> bool:
        """Validate image input."""
        return (image_path.exists() and 
                image_path.suffix.lower() in ProcessingConfig().supported_image_formats)
    
    def set_strategy(self, strategy: ExtractionStrategy) -> None:
        """Change the extraction strategy."""
        self._strategy = strategy


class BorderRefiner(ImageProcessor):
    """Processor for refining image borders."""
    
    def process(self, image_path: Path, output_dir: Path, config: ProcessingConfig, observer: Optional[ProgressObserver] = None) -> Optional[ImageMetadata]:
        """Refine image borders by removing empty space."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            gray = cv2.cvtColor(img, TABLE_EXTRACTION_COLORSPACE_CONVERSION)
            
            # Find content boundaries
            min_x = min_y = config.search_window_size
            height, width = gray.shape
            
            # Search for content in Y direction (with bounds checking)
            y_start = min(config.y_offset, height - 1)
            y_end = min(config.search_window_size + config.y_offset, height)
            x_end = min(config.search_window_size, width)
            
            for y in range(y_start, y_end):
                for x in range(x_end):
                    if gray[y, x] < config.refinement_threshold:
                        min_y = min(min_y, y)
                        break
            
            # Search for content in X direction (with bounds checking)
            x_start = min(config.x_offset, width - 1)
            x_end = min(config.search_window_size + config.x_offset, width)
            y_end = min(config.search_window_size, height)
            
            for x in range(x_start, x_end):
                for y in range(y_end):
                    if gray[y, x] < config.refinement_threshold:
                        min_x = min(min_x, x)
                        break
            
            # Crop image
            refined_img = img[min_y:, min_x:]
            
            output_path = output_dir / f"refined_{image_path.name}"
            cv2.imwrite(str(output_path), refined_img)
            
            return ImageMetadata(
                source_path=image_path,
                output_path=output_path,
                width=refined_img.shape[1],
                height=refined_img.shape[0],
                format=image_path.suffix
            )
            
        except Exception as e:
            raise ProcessingStageError(f"Border refinement failed: {e}") from e
    
    def validate_input(self, image_path: Path) -> bool:
        """Validate image input."""
        return (image_path.exists() and 
                image_path.suffix.lower() in ProcessingConfig().supported_image_formats)


class ConsoleProgressObserver:
    """Console-based progress observer."""
    
    def __init__(self, verbose: int = 0):
        self._verbose = verbose
    
    def on_stage_started(self, stage: ProcessingStage, message: str) -> None:
        """Log stage start."""
        if self._verbose:
            print(f"🔄 {stage.value.upper()}: {message}")
        logging.info(f"Stage started: {stage.value} - {message}")
    
    def on_stage_completed(self, stage: ProcessingStage, message: str) -> None:
        """Log stage completion."""
        if self._verbose:
            print(f"✅ {stage.value.upper()}: {message}")
        logging.info(f"Stage completed: {stage.value} - {message}")
    
    def on_progress_update(self, current: int, total: int, message: str) -> None:
        """Log progress update."""
        percentage = (current / total) * 100 if total > 0 else 0
        if self._verbose:
            print(f"📊 Progress: {current}/{total} ({percentage:.1f}%) - {message}")
        logging.debug(f"Progress: {current}/{total} - {message}")
    
    def on_error(self, stage: ProcessingStage, error: Exception) -> None:
        """Log error."""
        print(f"❌ ERROR in {stage.value}: {error}")
        logging.error(f"Error in {stage.value}: {error}", exc_info=True)


class TableSegmentationPipeline:
    """
    Main orchestrator for the table segmentation process.
    
    This class coordinates all processing stages and manages the overall workflow.
    It follows the Template Method pattern for the processing pipeline.
    """
    
    def __init__(
        self, 
        config: Optional[ProcessingConfig] = None,
        observer: Optional[ProgressObserver] = None
    ):
        self._config = config or ProcessingConfig()
        self._observer = observer or ConsoleProgressObserver()
        self._results: Dict[str, Any] = {}
    
    def process(
        self, 
        input_path: Path, 
        output_base_dir: Path, 
        options: ProcessingOptions
    ) -> Dict[str, Any]:
        """
        Execute the complete table segmentation pipeline.
        
        Args:
            input_path: Path to input PDF file or directory
            output_base_dir: Base directory for outputs
            options: Processing options
            
        Returns:
            Dictionary containing processing results and metadata
        """
        try:
            # Initialize
            self._initialize_processing(input_path, output_base_dir, options)
            
            # Execute processing stages
            if options.extract_images:
                self._extract_images_stage()
            
            if options.extract_tables:
                self._extract_tables_stage(options.extraction_method)
            
            if options.refine_borders:
                self._refine_borders_stage()
            
            # Finalize
            self._finalize_processing()
            
            return self._results
            
        except Exception as e:
            self._observer.on_error(ProcessingStage.COMPLETION, e)
            raise
    
    def _initialize_processing(
        self, 
        input_path: Path, 
        output_base_dir: Path, 
        options: ProcessingOptions
    ) -> None:
        """Initialize the processing pipeline."""
        self._observer.on_stage_started(
            ProcessingStage.INITIALIZATION, 
            f"Initializing processing for {input_path}"
        )
        
        # Validate inputs
        if not input_path.exists():
            raise InvalidInputError(f"Input path does not exist: {input_path}")
        
        # Setup directory structure
        self._input_path = input_path
        self._output_base_dir = output_base_dir
        self._options = options
        
        # Create output directories
        self._setup_output_directories()
        
        # Initialize results
        self._results = {
            'input_path': str(input_path),
            'output_base_dir': str(output_base_dir),
            'options': options,
            'processed_files': [],
            'errors': [],
            'processing_time': 0.0
        }
        
        self._observer.on_stage_completed(
            ProcessingStage.INITIALIZATION,
            "Initialization completed successfully"
        )
    
    def _setup_output_directories(self) -> None:
        """Create necessary output directories."""
        directories = {
            'images': self._output_base_dir / SUFFIX_EXTRACTED_IMAGES_FOLDER,
            'tables': self._output_base_dir / SUFFIX_EXTRACTED_TABLES_FOLDER, 
            'refined': self._output_base_dir / SUFFIX_EXTRACTED_REFINED_TABLE_BORDERS_FOLDER
        }
        
        for dir_type, dir_path in directories.items():
            dir_path.mkdir(parents=True, exist_ok=True)
            setattr(self, f"_{dir_type}_dir", dir_path)
    
    def _extract_images_stage(self) -> None:
        """Execute image extraction stage."""
        self._observer.on_stage_started(
            ProcessingStage.IMAGE_EXTRACTION,
            "Starting image extraction from PDF"
        )
        
        extractor = ImageExtractor()
        
        if self._input_path.is_file() and self._input_path.suffix.lower() == '.pdf':
            # Process single PDF
            self._observer.on_progress_update(
                1, 1,
                f"📄 Processing PDF: {self._input_path.name}"
            )
            
            # Passer l'observer au processeur pour logger chaque page
            images_metadata = extractor.process(self._input_path, self._images_dir, self._config, self._observer)
            self._results['extracted_images'] = [str(meta.output_path) for meta in images_metadata]
        else:
            # Handle directory of images (copy them to images directory)
            self._copy_existing_images()
        
        self._observer.on_stage_completed(
            ProcessingStage.IMAGE_EXTRACTION,
            f"Extracted {len(self._results.get('extracted_images', []))} images"
        )
    
    def _copy_existing_images(self) -> None:
        """Copy existing images from input directory."""
        copied_images = []
        
        # Compter les images à copier
        image_files = [f for f in self._input_path.iterdir() 
                      if f.suffix.lower() in self._config.supported_image_formats]
        
        self._observer.on_progress_update(
            0, len(image_files),
            f"Found {len(image_files)} images to copy"
        )
        
        for i, file_path in enumerate(image_files):
            self._observer.on_progress_update(
                i + 1, len(image_files),
                f"🖼️ Copying image: {file_path.name}"
            )
            
            dst_path = self._images_dir / file_path.name
            # Simple copy using cv2
            img = cv2.imread(str(file_path))
            if img is not None:
                cv2.imwrite(str(dst_path), img)
                copied_images.append(str(dst_path))
                self._observer.on_progress_update(
                    i + 1, len(image_files),
                    f"✅ Copied successfully: {file_path.name}"
                )
        
        self._results['extracted_images'] = copied_images
    
    def _extract_tables_stage(self, method: ExtractionMethod) -> None:
        """Execute table extraction stage."""
        self._observer.on_stage_started(
            ProcessingStage.TABLE_EXTRACTION,
            f"Starting table extraction using {method.value}"
        )
        
        # Get extraction strategy
        strategy = ExtractionStrategyFactory.create_strategy(method)
        extractor = TableExtractor(strategy)
        
        # Process all images
        image_paths = [Path(p) for p in self._results.get('extracted_images', [])]
        processed_tables = []
        
        for i, image_path in enumerate(image_paths):
            try:
                self._observer.on_progress_update(
                    i + 1, len(image_paths),
                    f"📋 Processing table extraction: {image_path.name}"
                )
                
                metadata = extractor.process(image_path, self._tables_dir, self._config, self._observer)
                if metadata:
                    processed_tables.append(str(metadata.output_path))
                    self._observer.on_progress_update(
                        i + 1, len(image_paths),
                        f"✅ Table extracted: {Path(metadata.output_path).name}"
                    )
                else:
                    self._observer.on_progress_update(
                        i + 1, len(image_paths),
                        f"⚠️ No table found in: {image_path.name}"
                    )
                
            except Exception as e:
                error_msg = f"Failed to process {image_path.name}: {e}"
                self._results['errors'].append(error_msg)
                self._observer.on_error(ProcessingStage.TABLE_EXTRACTION, e)
                self._observer.on_progress_update(
                    i + 1, len(image_paths),
                    f"❌ Failed to extract table from: {image_path.name}"
                )
        
        self._results['extracted_tables'] = processed_tables
        
        self._observer.on_stage_completed(
            ProcessingStage.TABLE_EXTRACTION,
            f"Processed {len(processed_tables)} tables"
        )
    
    def _refine_borders_stage(self) -> None:
        """Execute border refinement stage."""
        self._observer.on_stage_started(
            ProcessingStage.BORDER_REFINEMENT,
            "Starting border refinement"
        )
        
        refiner = BorderRefiner()
        table_paths = [Path(p) for p in self._results.get('extracted_tables', [])]
        refined_images = []
        
        for i, table_path in enumerate(table_paths):
            try:
                self._observer.on_progress_update(
                    i + 1, len(table_paths),
                    f"🔧 Refining borders: {table_path.name}"
                )
                
                metadata = refiner.process(table_path, self._refined_dir, self._config, self._observer)
                if metadata:
                    refined_images.append(str(metadata.output_path))
                    self._observer.on_progress_update(
                        i + 1, len(table_paths),
                        f"✅ Borders refined: {Path(metadata.output_path).name}"
                    )
                else:
                    self._observer.on_progress_update(
                        i + 1, len(table_paths),
                        f"⚠️ No refinement applied: {table_path.name}"
                    )
                
            except Exception as e:
                error_msg = f"Failed to refine {table_path.name}: {e}"
                self._results['errors'].append(error_msg)
                self._observer.on_error(ProcessingStage.BORDER_REFINEMENT, e)
                self._observer.on_progress_update(
                    i + 1, len(table_paths),
                    f"❌ Failed to refine: {table_path.name}"
                )
        
        self._results['refined_images'] = refined_images
        
        self._observer.on_stage_completed(
            ProcessingStage.BORDER_REFINEMENT,
            f"Refined {len(refined_images)} images"
        )

    
    def _extract_page_number(self, filename: str) -> int:
        """Extract page number from filename for sorting."""
        try:
            # Look for patterns like 'page_1.png' or 'image_1.jpg'
            parts = filename.split('_')
            for part in parts:
                if part.isdigit():
                    return int(part)
            return 0
        except:
            return 0
    
    def _finalize_processing(self) -> None:
        """Finalize the processing pipeline."""
        self._observer.on_stage_started(
            ProcessingStage.COMPLETION,
            "Finalizing processing results"
        )
        
        # Calculate summary statistics
        total_processed = (
            len(self._results.get('extracted_images', [])) +
            len(self._results.get('extracted_tables', [])) +
            len(self._results.get('refined_images', [])) +
            len(self._results.get('fused_images', []))
        )
        
        self._results['summary'] = {
            'total_processed_files': total_processed,
            'total_errors': len(self._results['errors']),
            'success_rate': (total_processed / (total_processed + len(self._results['errors'])) * 100) if total_processed > 0 else 0
        }
        
        success_msg = (
            f"Processing completed successfully. "
            f"Processed {total_processed} files with {len(self._results['errors'])} errors."
        )
        
        self._observer.on_stage_completed(ProcessingStage.COMPLETION, success_msg)


class PathValidator:
    """Utility class for path validation."""
    
    @staticmethod
    def validate_pdf_path(path: Path) -> bool:
        """Validate PDF file path."""
        return path.exists() and path.suffix.lower() == '.pdf'
    
    @staticmethod
    def validate_directory_path(path: Path) -> bool:
        """Validate directory path."""
        return path.exists() and path.is_dir()
    
    @staticmethod
    def validate_image_path(path: Path) -> bool:
        """Validate image file path."""
        supported_formats = ProcessingConfig().supported_image_formats
        return path.exists() and path.suffix.lower() in supported_formats


@contextmanager
def processing_timer(verbose: int = 0):
    """Context manager for timing processing operations."""
    import time
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        completion_msg = f"Processing completed in {end_time - start_time:.2f} seconds"
        if verbose:
            print(completion_msg)
        logging.info(completion_msg)


class TableProcessor:
    """
    Main interface for table processing operations.
    
    This class provides a simplified interface for users while maintaining
    the flexibility of the underlying architecture.
    """
    
    def __init__(
        self, 
        config: Optional[ProcessingConfig] = None,
        verbose: int = 0,
        observer: Optional[ProgressObserver] = None
    ):
        """
        Initialize the table processor.
        
        Args:
            config: Configuration for processing operations
            verbose: Whether to show detailed progress information
            observer: Optional custom progress observer
        """
        self._config = config or ProcessingConfig()
        self._observer = observer or ConsoleProgressObserver(verbose=verbose)
        self._pipeline = TableSegmentationPipeline(self._config, self._observer)
        
        # Compteurs et statistiques
        self._total_documents = 0
        self._processed_documents = 0
        self._successful_documents = 0
        self._failed_documents = 0
        self._session_start_time = None
    
    def process_pdf(
        self,
        pdf_path: str | Path,
        output_dir: str | Path,
        extract_images: bool = True,
        extract_tables: bool = True,
        refine_borders: bool = False,
        extraction_method: ExtractionMethod = ExtractionMethod.LINE_DETECTION
    ) -> Dict[str, Any]:
        """
        Process a PDF file for table extraction.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Output directory for results
            extract_images: Whether to extract images from PDF
            extract_tables: Whether to extract tables from images
            refine_borders: Whether to refine table borders
            extraction_method: Method to use for table extraction
            
        Returns:
            Dictionary containing processing results
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        
        # Initialiser le compteur de session si c'est le premier document
        if self._session_start_time is None:
            self._session_start_time = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._observer.on_stage_started(
                ProcessingStage.INITIALIZATION,
                f"📚 [SESSION START] {timestamp}"
            )
        
        # Incrémenter les compteurs
        self._total_documents += 1
        doc_start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Logs de début de traitement
        self._observer.on_stage_started(
            ProcessingStage.PDF_LOADING,
            f"📄 [PDF LOAD] {timestamp} - Document {self._processed_documents + 1}: {pdf_path.name}"
        )
        
        # Validate inputs
        if not PathValidator.validate_pdf_path(pdf_path):
            self._failed_documents += 1
            self._observer.on_error(
                ProcessingStage.PDF_LOADING,
                Exception(f"❌ [PDF ERROR] {timestamp} - Invalid PDF path: {pdf_path}")
            )
            raise InvalidInputError(f"Invalid PDF path: {pdf_path}")
        
        try:
            # Log du début de traitement détaillé
            self._observer.on_stage_started(
                ProcessingStage.INITIALIZATION,
                f"🔧 [PROCESSING] Options: images={extract_images}, tables={extract_tables}, borders={refine_borders}"
            )
            
            # Create processing options
            options = ProcessingOptions(
                extract_images=extract_images,
                extract_tables=extract_tables,
                refine_borders=refine_borders,
                extraction_method=extraction_method
            )
            
            with processing_timer(verbose=self._observer.verbose):
                result = self._pipeline.process(pdf_path, output_dir, options)
                
                # Logs de fin de traitement réussi
                self._processed_documents += 1
                self._successful_documents += 1
                doc_end_time = time.time()
                doc_duration = doc_end_time - doc_start_time
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                self._observer.on_stage_completed(
                    ProcessingStage.COMPLETION,
                    f"✅ [PDF SUCCESS] {timestamp} - {pdf_path.name} processed in {doc_duration:.2f}s ({self._processed_documents}/{self._total_documents} completed)"
                )
                
                return result
                
        except Exception as e:
            # Logs d'erreur de traitement
            self._processed_documents += 1
            self._failed_documents += 1
            doc_end_time = time.time()
            doc_duration = doc_end_time - doc_start_time
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            self._observer.on_error(
                ProcessingStage.COMPLETION,
                Exception(f"❌ [PDF FAILED] {timestamp} - {pdf_path.name} failed after {doc_duration:.2f}s: {str(e)} ({self._processed_documents}/{self._total_documents} completed)")
            )
            
            raise
    
    def process_images(
        self,
        images_dir: str | Path,
        output_dir: str | Path,
        extract_tables: bool = True,
        refine_borders: bool = False,
        extraction_method: ExtractionMethod = ExtractionMethod.LINE_DETECTION
    ) -> Dict[str, Any]:
        """
        Process a directory of images for table extraction.
        
        Args:
            images_dir: Directory containing images
            output_dir: Output directory for results
            extract_tables: Whether to extract tables from images
            refine_borders: Whether to refine table borders
            extraction_method: Method to use for table extraction
            
        Returns:
            Dictionary containing processing results
        """
        images_dir = Path(images_dir)
        output_dir = Path(output_dir)
        
        # Initialiser le compteur de session si c'est le premier document
        if self._session_start_time is None:
            self._session_start_time = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._observer.on_stage_started(
                ProcessingStage.INITIALIZATION,
                f"📚 [SESSION START] {timestamp}"
            )
        
        # Incrémenter les compteurs
        self._total_documents += 1
        doc_start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Logs de début de traitement
        self._observer.on_stage_started(
            ProcessingStage.IMAGE_EXTRACTION,
            f"🖼️ [IMAGES LOAD] {timestamp} - Document {self._processed_documents + 1}: {images_dir.name}"
        )
        
        # Validate inputs
        if not PathValidator.validate_directory_path(images_dir):
            self._failed_documents += 1
            self._observer.on_error(
                ProcessingStage.IMAGE_EXTRACTION,
                Exception(f"❌ [IMAGES ERROR] {timestamp} - Invalid images directory: {images_dir}")
            )
            raise InvalidInputError(f"Invalid images directory: {images_dir}")
        
        try:
            # Log du début de traitement détaillé
            self._observer.on_stage_started(
                ProcessingStage.INITIALIZATION,
                f"🔧 [PROCESSING] Options: tables={extract_tables}, borders={refine_borders}"
            )
            
            # Create processing options (no image extraction needed)
            options = ProcessingOptions(
                extract_images=True,  # Will copy existing images
                extract_tables=extract_tables,
                refine_borders=refine_borders,
                extraction_method=extraction_method
            )
            
            with processing_timer(verbose=self._observer.verbose):
                result = self._pipeline.process(images_dir, output_dir, options)
                
                # Logs de fin de traitement réussi
                self._processed_documents += 1
                self._successful_documents += 1
                doc_end_time = time.time()
                doc_duration = doc_end_time - doc_start_time
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                self._observer.on_stage_completed(
                    ProcessingStage.COMPLETION,
                    f"✅ [IMAGES SUCCESS] {timestamp} - {images_dir.name} processed in {doc_duration:.2f}s ({self._processed_documents}/{self._total_documents} completed)"
                )
                
                return result
                
        except Exception as e:
            # Logs d'erreur de traitement
            self._processed_documents += 1
            self._failed_documents += 1
            doc_end_time = time.time()
            doc_duration = doc_end_time - doc_start_time
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            self._observer.on_error(
                ProcessingStage.COMPLETION,
                Exception(f"❌ [IMAGES FAILED] {timestamp} - {images_dir.name} failed after {doc_duration:.2f}s: {str(e)} ({self._processed_documents}/{self._total_documents} completed)")
            )
            
            raise
    
    def get_available_extraction_methods(self) -> List[str]:
        """Get list of available extraction methods."""
        methods = ExtractionStrategyFactory.get_available_methods()
        return [method.value for method in methods]
    
    def create_custom_config(
        self,
        zoom_factor: float = 300.0 / 72.0,
        binarization_threshold: int = 128,
        **kwargs
    ) -> ProcessingConfig:
        """
        Create a custom configuration object.
        
        Args:
            zoom_factor: Zoom factor for image extraction
            binarization_threshold: Threshold for image binarization
            **kwargs: Additional configuration parameters
            
        Returns:
            Custom ProcessingConfig object
        """
        config_dict = {
            'zoom_factor': zoom_factor,
            'binarization_threshold': binarization_threshold,
            **kwargs
        }
        
        # Filter out invalid parameters
        valid_params = {k: v for k, v in config_dict.items() 
                       if hasattr(ProcessingConfig, k)}
        
        return ProcessingConfig(**valid_params)

def main():
    """Example usage of the refactored table processor."""
    
    # Logging est configuré par le CLI ou par défaut
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
    
    try:
        # Create processor with custom configuration
        config = ProcessingConfig(
            zoom_factor=300.0 / 72.0,
            binarization_threshold=120
        )
        
        processor = TableProcessor(config=config, verbose=0)
        
        # Example: Process a PDF file
        results = processor.process_pdf(
            pdf_path="./inputs/sample.pdf",
            output_dir="./outputs/sample_output",
            extract_images=True,
            extract_tables=True,
            refine_borders=True,
            extraction_method=ExtractionMethod.LINE_DETECTION
        )
        
        # Print results
        print("\n" + "="*50)
        print("PROCESSING RESULTS")
        print("="*50)
        print(f"Input: {results['input_path']}")
        print(f"Output: {results['output_base_dir']}")
        print(f"Success Rate: {results['summary']['success_rate']:.1f}%")
        print(f"Total Files Processed: {results['summary']['total_processed_files']}")
        print(f"Errors: {results['summary']['total_errors']}")
        
        # Logs des résultats
        logging.info("PROCESSING RESULTS")
        logging.info(f"Input: {results['input_path']}")
        logging.info(f"Output: {results['output_base_dir']}")
        logging.info(f"Success Rate: {results['summary']['success_rate']:.1f}%")
        logging.info(f"Total Files Processed: {results['summary']['total_processed_files']}")
        logging.info(f"Errors: {results['summary']['total_errors']}")
        
        if results['errors']:
            print("\nErrors encountered:")
            logging.info("Errors encountered:")
            for error in results['errors']:
                print(f"  - {error}")
                logging.error(f"Error: {error}")
        
    except Exception as e:
        print(f"Processing failed: {e}")
        logging.error(f"Processing failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()