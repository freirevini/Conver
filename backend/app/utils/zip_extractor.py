"""
ZIP Extractor utility for KNIME workflows.
Handles extraction of .knwf and .zip workflow archives.
"""
import os
import zipfile
import shutil
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ZipExtractor:
    """Extracts KNIME workflow archives."""
    
    def extract(self, zip_path: str, extract_dir: Optional[str] = None) -> str:
        """
        Extract ZIP/KNWF file to a directory.
        
        Args:
            zip_path: Path to the ZIP/KNWF file
            extract_dir: Optional extraction directory. If not provided,
                        extracts to a subdirectory next to the ZIP file.
        
        Returns:
            Path to the extracted workflow directory containing workflow.knime
        """
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Archive not found: {zip_path}")
        
        # Determine extraction directory
        if extract_dir is None:
            extract_dir = os.path.splitext(zip_path)[0] + "_extracted"
        
        # Create extraction directory
        os.makedirs(extract_dir, exist_ok=True)
        
        logger.info(f"Extracting {zip_path} to {extract_dir}")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        except zipfile.BadZipFile:
            raise ValueError(f"Invalid ZIP file: {zip_path}")
        
        # Find the workflow.knime file to determine workflow root
        workflow_root = self._find_workflow_root(extract_dir)
        
        if workflow_root is None:
            raise ValueError(
                f"No valid KNIME workflow found in archive. "
                f"Expected workflow.knime or settings.xml files."
            )
        
        logger.info(f"Workflow root found: {workflow_root}")
        return workflow_root
    
    def _find_workflow_root(self, extract_dir: str) -> Optional[str]:
        """
        Find the root directory of the KNIME workflow.
        
        Searches for workflow.knime or directories containing node settings.
        KNIME workflows can have nested structures like:
        archive/document/WorkflowName/workflow.knime
        """
        # Walk through directory tree to find workflow.knime
        for root, dirs, files in os.walk(extract_dir):
            # Check for workflow.knime (main workflow file)
            if 'workflow.knime' in files:
                # This could be a metanode, check if parent has workflow.knime too
                parent = os.path.dirname(root)
                if 'workflow.knime' not in os.listdir(parent):
                    # This is a top-level workflow or main metanode
                    return root
            
            # Check for directories containing settings.xml (node directories)
            for d in dirs:
                settings_path = os.path.join(root, d, 'settings.xml')
                if os.path.exists(settings_path):
                    # Found node directories, this is the workflow root
                    return root
        
        # Fallback: return the first directory with any XML files
        for root, dirs, files in os.walk(extract_dir):
            xml_files = [f for f in files if f.endswith('.xml')]
            if xml_files:
                return root
        
        return None
    
    def cleanup(self, extract_dir: str) -> None:
        """Remove extracted directory."""
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
            logger.info(f"Cleaned up: {extract_dir}")
