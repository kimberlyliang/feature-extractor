#%%
import os
import pandas as pd
import numpy as np
import h5py
from process_ieeg import IEEGClipProcessor
from IPython import embed
from pathlib import Path
from features_univariate_utils import in_parallel, catch22_single_series, compute_psd_all_channels_parallel, compute_entropy_single_series
# from fooof import FOOOF
import time
from functools import lru_cache
import logging
from typing import Dict, List, Tuple, Optional
import warnings
import re

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

def find_h5_file(input_dir: Path) -> Tuple[Path, str]:
    """Find the H5 file in the input directory and extract subject ID.
    
    Args:
        input_dir: Path to input directory or file
        
    Returns:
        Tuple of (h5_file_path, subject_id)
    """
    logger.info(f"Searching for H5 file in: {input_dir}")
    
    # Define possible filenames
    possible_filenames = ['interictal_ieeg_processed.h5', 'interictal_ieeg_wake_processed.h5']
    
    # First check if input_dir is one of the files
    if input_dir.is_file() and input_dir.name in possible_filenames:
        h5_file = input_dir
    else:
        # Search recursively for any of the possible files
        h5_files = []
        for filename in possible_filenames:
            found_files = list(input_dir.rglob(filename))
            if found_files:
                logger.info(f"Found {len(found_files)} files matching {filename}")
                h5_files.extend(found_files)
        
        logger.info(f"Total H5 files found: {len(h5_files)}")
        
        if not h5_files:
            # If no files found, try searching in subdirectories
            for subdir in input_dir.iterdir():
                if subdir.is_dir():
                    logger.info(f"Checking subdirectory: {subdir}")
                    for filename in possible_filenames:
                        found_files = list(subdir.rglob(filename))
                        if found_files:
                            logger.info(f"Found {len(found_files)} files matching {filename} in {subdir}")
                            h5_files.extend(found_files)
                            break
                    if h5_files:
                        break
            
            if not h5_files:
                raise FileNotFoundError(f"No H5 files found in {input_dir} or its subdirectories")
        
        if len(h5_files) > 1:
            logger.warning(f"Found multiple H5 files: {h5_files}")
            logger.info("Using the first file found")
        
        h5_file = h5_files[0]
        logger.info(f"Using file: {h5_file}")
    
    # Extract subject ID from path
    try:
        # Try to find sub-* pattern in the path
        subject_match = re.search(r'sub-[A-Za-z0-9]+', str(h5_file))
        if subject_match:
            subject_id = subject_match.group(0)
            logger.info(f"Extracted subject ID from path: {subject_id}")
        else:
            # Use parent directory name as fallback
            subject_id = h5_file.parent.name
            logger.info(f"No subject ID found in path, using parent directory name: {subject_id}")
    except Exception as e:
        logger.warning(f"Error extracting subject ID: {e}")
        subject_id = h5_file.parent.name
    
    logger.info(f"Final subject ID: {subject_id}")
    return h5_file, subject_id

#%%
class UnivariateFeatures(IEEGClipProcessor):
    def __init__(self, input_dir: Path, chunk_size: int = 1000):
        logger.info(f"Initializing UnivariateFeatures with input_dir: {input_dir}")
        super().__init__()
        self.chunk_size = chunk_size
        
        # Find H5 file and get subject ID
        self.ieeg_processed, self.subject_id = find_h5_file(input_dir)
        logger.info(f"Found H5 file for subject {self.subject_id} at: {self.ieeg_processed}")
        
        # Set output directory to be the same as the input file's directory
        # self.output_dir = self.ieeg_processed.parent
        self.output_dir = output_base_dir
        logger.info(f"Output directory set to: {self.output_dir}")
        
        # Initialize data attributes
        self._ieeg_processed_bipolar = None
        self._coordinates = None
        self._sampling_frequency = None
        self._psd_df = None
        
        # Load data
        self._load_data()
    
    @property
    def ieeg_processed_bipolar(self) -> pd.DataFrame:
        """Lazy loading of iEEG data"""
        logger.debug("Accessing ieeg_processed_bipolar property")
        if self._ieeg_processed_bipolar is None:
            logger.info("iEEG data not loaded, loading now...")
            self._load_data()
        return self._ieeg_processed_bipolar
    
    @property
    def coordinates(self) -> pd.DataFrame:
        """Lazy loading of coordinates"""
        logger.debug("Accessing coordinates property")
        if self._coordinates is None:
            logger.info("Coordinates not loaded, loading now...")
            self._load_data()
        return self._coordinates
    
    @property
    def sampling_frequency(self) -> float:
        """Lazy loading of sampling frequency"""
        if self._sampling_frequency is None:
            logger.info("Sampling frequency not loaded, loading now...")
            self._load_data()
        return self._sampling_frequency
    
    @property
    def psd_df(self) -> pd.DataFrame:
        """Lazy loading of PSD data"""
        logger.debug("Accessing psd_df property")
        if self._psd_df is None:
            logger.info("PSD data not computed, computing now...")
            self._psd_df = self._compute_psd()
        return self._psd_df
    
    def _load_data(self) -> None:
        """Load and initialize all data"""
        logger.info("Starting data loading process...")
        start_time = time.time()
        
        if not self.ieeg_processed.exists():
            raise FileNotFoundError(f"No iEEG processed file found at {self.ieeg_processed}")
        
        logger.info(f"Opening H5 file: {self.ieeg_processed}")
        with h5py.File(self.ieeg_processed, 'r') as f:
            logger.info("Loading coordinates data...")
            coordinates_data = f['/bipolar_montage/coordinates']
            logger.info("Loading iEEG data...")
            ieeg_data = f['/bipolar_montage/ieeg']
            
            # Load data in chunks to save memory
            logger.info(f"Loading iEEG data in chunks of size {self.chunk_size}...")
            logger.info(f"Total data shape: {ieeg_data.shape}")
            chunks = []
            total_chunks = (ieeg_data.shape[0] + self.chunk_size - 1) // self.chunk_size
            logger.info(f"Will process {total_chunks} chunks")
            
            for i in range(0, ieeg_data.shape[0], self.chunk_size):
                chunk_num = i // self.chunk_size + 1
                logger.info(f"Loading chunk {chunk_num}/{total_chunks}...")
                chunk = ieeg_data[i:i + self.chunk_size]
                logger.info(f"Chunk shape: {chunk.shape}")
                chunks.append(pd.DataFrame(chunk, columns=ieeg_data.attrs['channels_labels']))
            
            logger.info("Concatenating chunks...")
            self._ieeg_processed_bipolar = pd.concat(chunks, axis=0)
            logger.info(f"Final DataFrame shape: {self._ieeg_processed_bipolar.shape}")
            
            self._sampling_frequency = ieeg_data.attrs['sampling_rate']
            logger.info(f"Sampling frequency: {self._sampling_frequency} Hz")
            
            # Load coordinates data
            logger.info("Processing coordinates data...")
            logger.info(f"Available coordinate attributes: {list(coordinates_data.attrs.keys())}")
            orig_labels = coordinates_data.attrs['original_labels'] if 'original_labels' in coordinates_data.attrs else None
            self._coordinates = pd.concat([
                pd.DataFrame(coordinates_data, columns=['x', 'y', 'z'], index=coordinates_data.attrs['labels']),
                pd.DataFrame(orig_labels, columns=['orig_labels'], index=coordinates_data.attrs['labels']),
                pd.DataFrame(coordinates_data.attrs['roi'], columns=['roi'], index=coordinates_data.attrs['labels']),
                pd.DataFrame(coordinates_data.attrs['roiNum'], columns=['roiNum'], index=coordinates_data.attrs['labels']),
                pd.DataFrame(coordinates_data.attrs['spared'], columns=['spared'], index=coordinates_data.attrs['labels'])
            ], axis=1)
            logger.info(f"Coordinates DataFrame shape: {self._coordinates.shape}")
        
        end_time = time.time()
        logger.info(f"Data loading completed in {end_time - start_time:.2f} seconds")
    
    def _compute_psd(self) -> pd.DataFrame:
        """Compute PSD for all channels in chunks"""
        logger.info("Starting PSD computation...")
        start_time = time.time()
        
        logger.info(f"Input data shape: {self.ieeg_processed_bipolar.shape}")
        result = compute_psd_all_channels_parallel(self.ieeg_processed_bipolar, self.sampling_frequency)
        logger.info(f"PSD result shape: {result.shape}")
        
        end_time = time.time()
        logger.info(f"PSD computation completed in {end_time - start_time:.2f} seconds")
        return result
    
    def catch22_features(self) -> pd.DataFrame:
        """Calculate Catch22 features for all channels."""
        logger.info("Calculating Catch22 features...")
        start_time = time.time()
        
        # Get data
        data = self.ieeg_processed_bipolar
        channels = data.columns
        logger.info(f"Processing {len(channels)} channels...")
        
        # Process in smaller batches to avoid memory issues
        batch_size = 5  # Process 5 channels at a time
        results = []
        
        try:
            for i in range(0, len(channels), batch_size):
                batch_channels = channels[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(channels) + batch_size - 1)//batch_size}: channels {batch_channels[0]}-{batch_channels[-1]}")
                
                # Calculate features for this batch
                batch_results = in_parallel(
                    catch22_single_series,
                    [data[col].values for col in batch_channels],
                    n_jobs=min(2, len(batch_channels)),  # Limit to 2 parallel jobs
                    verbose=True,
                    timeout=60  # 1 minute timeout per batch
                )
                
                # Convert results to DataFrame
                feature_names = batch_results[0]['names']
                feature_values = np.array([result['values'] for result in batch_results])
                batch_df = pd.DataFrame(feature_values, index=batch_channels, columns=feature_names)
                results.append(batch_df)
                
                logger.info(f"Completed batch {i//batch_size + 1}")
                
                # Save intermediate results
                if len(results) > 0:
                    temp_df = pd.concat(results)
                    temp_path = self.output_dir / f"catch22_features_temp_{i//batch_size + 1}.csv"
                    temp_df.to_csv(temp_path)
                    logger.info(f"Saved intermediate results to {temp_path}")
        
        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user")
            if len(results) > 0:
                logger.info("Saving partial results...")
                result_df = pd.concat(results)
                result_df.to_csv(self.output_dir / "catch22_features_partial.csv")
                logger.info("Partial results saved")
            raise
        
        # Combine all results
        result_df = pd.concat(results)
        logger.info(f"Catch22 features calculated in {time.time() - start_time:.2f} seconds")
        return result_df
    
    def bandpower_features(self) -> pd.DataFrame:
        """Calculate bandpower features for all channels."""
        logger.info("Calculating bandpower features...")
        start_time = time.time()
        
        # Get PSD data
        psd_df = self.psd_df
        
        # Define frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        # Calculate bandpower for each band
        result_df = pd.DataFrame(index=psd_df.columns)
        for band_name, (low, high) in bands.items():
            logger.info(f"Processing {band_name} band ({low}-{high} Hz)...")
            mask = (psd_df.index >= low) & (psd_df.index <= high)
            result_df[band_name] = psd_df.loc[mask].mean()
        
        logger.info(f"Bandpower features calculated in {time.time() - start_time:.2f} seconds")
        return result_df
    
    def entropy_features(self) -> pd.Series:
        """Calculate entropy features for all channels."""
        logger.info("Calculating entropy features...")
        start_time = time.time()
        
        # Get data
        data = self.ieeg_processed_bipolar
        
        # Calculate entropy in parallel
        results = in_parallel(compute_entropy_single_series, [data[col].values for col in data.columns])
        
        # Convert results to Series
        entropy_series = pd.Series(results, index=data.columns, name='entropy')
        
        logger.info(f"Entropy features calculated in {time.time() - start_time:.2f} seconds")
        return entropy_series
    
    def save_features(self) -> None:
        """Save all calculated features to CSV files in the same directory as the input file."""
        logger.info(f"Saving features to {self.output_dir}")
        
        # Calculate features
        bandpower_df = self.bandpower_features()
        entropy_series = self.entropy_features()
        
        # Save each feature set in the same directory as the input file
        bandpower_df.to_csv(self.output_dir / "bandpower_features.csv")
        entropy_df = entropy_series.to_frame()
        entropy_df.to_csv(self.output_dir / "entropy_features.csv")
        logger.info(f"Saved bandpower and entropy features in {self.output_dir}")

#%%

if __name__ == "__main__":
    logger.info("Starting feature extraction process...")
    start_time = time.time()
    
    # Get base input directory from environment variable or default
    input_base_dir = Path(os.environ.get('INPUT_DIR', 'data/input'))
    output_base_dir = Path(os.environ.get('OUTPUT_DIR', 'data/output'))
    logger.info(f"Input directory: {input_base_dir}")
    logger.info(f"Output directory: {output_base_dir}")
    
    try:
        # Find and process the H5 file
        logger.info("Initializing feature calculator...")
        features_calculator = UnivariateFeatures(input_base_dir)
        
        # Calculate and save all features
        logger.info("Starting feature calculation and saving...")
        features_calculator.save_features()
        
        end_time = time.time()
        logger.info(f"All features calculated and saved for {features_calculator.subject_id}")
        logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}", exc_info=True)
        raise
