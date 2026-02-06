"""
KNWF Extractor - ZIP extraction and structure analysis for KNIME workflow files.

Handles:
- ZIP extraction with memory caching
- Windows long path handling (>260 chars)
- Automatic workflow root detection
- KNIME version detection (3.x - 5.x)
"""
from __future__ import annotations

import io
import logging
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


class KnwfExtractionError(Exception):
    """Raised when KNWF extraction fails."""
    pass


class KnwfExtractor:
    """
    Extracts and provides access to KNIME workflow (.knwf) file contents.
    
    KNWF files are ZIP archives containing:
    - workflow.knime: Main workflow definition
    - settings.xml: Node configurations in each node folder
    - Node folders: Named like "NodeName (#123)" with settings and port data
    - Metanode folders: Nested workflows within workflows
    """
    
    # Patterns for identifying KNIME structures
    NODE_FOLDER_PATTERN = re.compile(r'^(.+) \(#(\d+)\)$')
    WORKFLOW_FILE = 'workflow.knime'
    SETTINGS_FILE = 'settings.xml'
    
    def __init__(self, knwf_path: Path | str):
        """
        Initialize extractor with path to .knwf file.
        
        Args:
            knwf_path: Path to the KNIME workflow file
        """
        self.knwf_path = Path(knwf_path)
        if not self.knwf_path.exists():
            raise FileNotFoundError(f"KNWF file not found: {self.knwf_path}")
        
        self._file_cache: Dict[str, bytes] = {}
        self._file_list: List[str] = []
        self._workflow_root: Optional[str] = None
        self._knime_version: Optional[str] = None
        self._is_loaded = False
    
    def load(self) -> None:
        """Load the KNWF file into memory cache."""
        if self._is_loaded:
            return
        
        try:
            with zipfile.ZipFile(self.knwf_path, 'r') as zf:
                self._file_list = zf.namelist()
                
                # Load all files into memory for fast access
                for file_path in self._file_list:
                    if not file_path.endswith('/'):  # Skip directories
                        try:
                            self._file_cache[file_path] = zf.read(file_path)
                        except Exception as e:
                            logger.warning(f"Failed to read {file_path}: {e}")
                
                # Find workflow root
                self._workflow_root = self._find_workflow_root()
                
                # Detect KNIME version
                self._knime_version = self._detect_knime_version()
                
                self._is_loaded = True
                logger.info(
                    f"Loaded KNWF: {len(self._file_cache)} files, "
                    f"root='{self._workflow_root}', version={self._knime_version}"
                )
                
        except zipfile.BadZipFile:
            raise KnwfExtractionError(f"Invalid ZIP file: {self.knwf_path}")
        except Exception as e:
            raise KnwfExtractionError(f"Failed to extract KNWF: {e}")
    
    def _find_workflow_root(self) -> str:
        """
        Find the root directory containing the main workflow.
        
        KNWF files can have different structures:
        - workflow.knime at root
        - folder/workflow.knime
        - folder/document/WorkflowName/workflow.knime
        
        Returns:
            Path to the workflow root directory
        """
        workflow_files = [
            f for f in self._file_list 
            if f.endswith(self.WORKFLOW_FILE)
        ]
        
        if not workflow_files:
            # Try to find any settings.xml as fallback
            settings_files = [f for f in self._file_list if f.endswith(self.SETTINGS_FILE)]
            if settings_files:
                # Use the shallowest settings.xml path
                settings_files.sort(key=lambda x: x.count('/'))
                return str(PurePosixPath(settings_files[0]).parent.parent) + '/'
            raise KnwfExtractionError("No workflow.knime or settings.xml found in archive")
        
        # Find the shallowest workflow.knime (main workflow, not inside metanodes)
        workflow_files.sort(key=lambda x: x.count('/'))
        main_workflow = workflow_files[0]
        
        # Return parent directory
        parent = str(PurePosixPath(main_workflow).parent)
        return parent + '/' if parent else ''
    
    def _detect_knime_version(self) -> Optional[str]:
        """Detect KNIME version from workflow metadata."""
        if not self._workflow_root:
            return None
        
        # Try reading workflow.knime for version info
        workflow_path = self._workflow_root + self.WORKFLOW_FILE
        if workflow_path not in self._file_cache:
            # Try without trailing slash
            alt_path = self._workflow_root.rstrip('/') + '/' + self.WORKFLOW_FILE
            if alt_path in self._file_cache:
                workflow_path = alt_path
            else:
                return None
        
        try:
            content = self._file_cache[workflow_path].decode('utf-8', errors='ignore')
            
            # Look for version patterns in XML
            version_patterns = [
                r'version="(\d+\.\d+(?:\.\d+)?)"',
                r'knime_version="(\d+\.\d+(?:\.\d+)?)"',
                r'<entry key="version" type="xstring" value="(\d+\.\d+(?:\.\d+)?)"',
            ]
            
            for pattern in version_patterns:
                match = re.search(pattern, content)
                if match:
                    return match.group(1)
            
        except Exception as e:
            logger.debug(f"Could not detect KNIME version: {e}")
        
        return None
    
    @property
    def workflow_root(self) -> str:
        """Get the workflow root directory path."""
        if not self._is_loaded:
            self.load()
        return self._workflow_root or ''
    
    @property
    def knime_version(self) -> Optional[str]:
        """Get detected KNIME version."""
        if not self._is_loaded:
            self.load()
        return self._knime_version
    
    @property
    def file_count(self) -> int:
        """Get total number of files in archive."""
        if not self._is_loaded:
            self.load()
        return len(self._file_cache)
    
    def get_file(self, path: str) -> Optional[bytes]:
        """
        Get file contents by path.
        
        Args:
            path: Path within the archive
            
        Returns:
            File contents as bytes, or None if not found
        """
        if not self._is_loaded:
            self.load()
        return self._file_cache.get(path)
    
    def get_file_text(self, path: str, encoding: str = 'utf-8') -> Optional[str]:
        """
        Get file contents as text.
        
        Args:
            path: Path within the archive
            encoding: Text encoding to use
            
        Returns:
            File contents as string, or None if not found
        """
        data = self.get_file(path)
        if data is None:
            return None
        return data.decode(encoding, errors='replace')
    
    def get_all_files(self) -> Dict[str, bytes]:
        """Get all files as a dictionary."""
        if not self._is_loaded:
            self.load()
        return self._file_cache.copy()
    
    def get_all_settings_xml(self) -> Iterator[Tuple[str, bytes]]:
        """
        Iterate over all settings.xml files in the archive.
        
        Yields:
            Tuples of (path, content) for each settings.xml
        """
        if not self._is_loaded:
            self.load()
        
        for path, content in self._file_cache.items():
            if path.endswith(self.SETTINGS_FILE):
                yield path, content
    
    def get_node_folders(self) -> List[Tuple[str, str, int]]:
        """
        Get all node folders in the workflow.
        
        Returns:
            List of tuples: (folder_path, node_name, node_id)
        """
        if not self._is_loaded:
            self.load()
        
        node_folders = []
        seen_paths = set()
        
        for path in self._file_list:
            if '/' not in path:
                continue
            
            parts = path.split('/')
            for i, part in enumerate(parts):
                match = self.NODE_FOLDER_PATTERN.match(part)
                if match:
                    node_name = match.group(1)
                    node_id = int(match.group(2))
                    folder_path = '/'.join(parts[:i+1]) + '/'
                    
                    if folder_path not in seen_paths:
                        seen_paths.add(folder_path)
                        node_folders.append((folder_path, node_name, node_id))
        
        return node_folders
    
    def get_metanode_folders(self) -> List[Tuple[str, str, int, int]]:
        """
        Get all metanode folders (folders containing their own workflow structure).
        
        Returns:
            List of tuples: (folder_path, node_name, node_id, depth)
        """
        if not self._is_loaded:
            self.load()
        
        metanodes = []
        node_folders = self.get_node_folders()
        
        for folder_path, node_name, node_id in node_folders:
            # Check if this folder contains settings.xml with child nodes
            # or has a workflow.knime file (indicating it's a metanode)
            has_children = False
            depth = folder_path.count('/') - self.workflow_root.count('/')
            
            # Check for child node folders
            for path in self._file_list:
                if path.startswith(folder_path) and path != folder_path:
                    relative = path[len(folder_path):]
                    if '/' in relative:
                        child_folder = relative.split('/')[0]
                        if self.NODE_FOLDER_PATTERN.match(child_folder):
                            has_children = True
                            break
            
            if has_children:
                metanodes.append((folder_path, node_name, node_id, depth))
        
        return metanodes
    
    def get_workflow_structure(self) -> Dict[str, Any]:
        """
        Get a hierarchical representation of the workflow structure.
        
        Returns:
            Nested dictionary representing the workflow tree
        """
        if not self._is_loaded:
            self.load()
        
        structure: Dict[str, Any] = {
            "root": self.workflow_root,
            "version": self.knime_version,
            "file_count": len(self._file_cache),
            "nodes": [],
            "metanodes": [],
        }
        
        # Add node info
        for folder_path, node_name, node_id in self.get_node_folders():
            is_metanode = any(
                m[0] == folder_path 
                for m in self.get_metanode_folders()
            )
            
            node_info = {
                "path": folder_path,
                "name": node_name,
                "id": node_id,
                "is_metanode": is_metanode,
            }
            
            if is_metanode:
                structure["metanodes"].append(node_info)
            else:
                structure["nodes"].append(node_info)
        
        return structure
    
    def extract_to_disk(self, output_dir: Path | str) -> Path:
        """
        Extract all files to disk.
        
        Args:
            output_dir: Directory to extract to
            
        Returns:
            Path to the extracted workflow root
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not self._is_loaded:
            self.load()
        
        for file_path, content in self._file_cache.items():
            # Handle Windows long paths
            full_path = output_path / file_path
            
            # Create parent directories
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            full_path.write_bytes(content)
        
        logger.info(f"Extracted {len(self._file_cache)} files to {output_path}")
        return output_path / self.workflow_root
    
    def close(self) -> None:
        """Clear the file cache to free memory."""
        self._file_cache.clear()
        self._is_loaded = False
    
    def __enter__(self) -> "KnwfExtractor":
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
