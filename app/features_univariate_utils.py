import logging
import os
from pathlib import Path
from joblib import Parallel, delayed
import pycatch22 as catch22
import scipy.signal
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# # Suppress FOOOF deprecation warning
# warnings.filterwarnings('ignore', category=DeprecationWarning, module='fooof')

def in_parallel(func, data: List[Any], verbose: bool = True, n_jobs: int = -1, timeout: int = 300) -> List[Any]:
    """Process data in parallel with progress tracking and error handling.
    
    Args:
        func: Function to apply to each item in data
        data: List of items to process
        verbose: Whether to show progress bar
        n_jobs: Number of parallel jobs (-1 for all cores)
        timeout: Maximum time in seconds to wait for each item
        
    Returns:
        List of results
    """
    if n_jobs == -1:
        n_jobs = min(os.cpu_count(), 4)  # Limit to 4 cores by default
    
    if verbose:
        logger.info(f"Processing {len(data)} items in parallel using {n_jobs} threads")
        data = tqdm(data, desc="Processing items", mininterval=1.0)  # Update progress bar every second
    
    try:
        results = Parallel(
            n_jobs=n_jobs,
            prefer="threads",
            batch_size=1,
            timeout=timeout,
            max_nbytes=None,  # Disable memory mapping
            require='sharedmem'  # Use shared memory for better performance
        )(
            delayed(func)(item) for item in data
        )
        return results
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        raise
    except Exception as e:
        logger.error(f"Error in parallel processing: {str(e)}")
        raise

def catch22_single_series(series: np.ndarray) -> Dict[str, Any]:
    """Calculate Catch22 features for a single time series.
    
    Args:
        series: Input time series
        
    Returns:
        Dictionary containing feature names and values
    """
    try:
        # Downsample if the series is too long
        if len(series) > 1000000:  # If more than 1 million points
            logger.warning(f"Downsampling series from {len(series)} to 1000000 points")
            series = scipy.signal.resample(series, 1000000)
        
        return catch22.catch22_all(series, catch24=True)
    except Exception as e:
        logger.error(f"Error calculating Catch22 features: {str(e)}")
        raise

def compute_psd(data: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density for a single time series.
    
    Args:
        data: Input time series
        fs: Sampling frequency
        
    Returns:
        Tuple of (frequencies, power spectral density)
    """
    return scipy.signal.welch(
        data, 
        fs=fs, 
        window='hamming', 
        nperseg=fs * 2, 
        noverlap=fs, 
        scaling='density'
    )

def compute_psd_all_channels_parallel(data: pd.DataFrame, fs: float, chunk_size: int = 1000) -> pd.DataFrame:
    """Compute PSD for all channels in parallel with memory-efficient processing.
    
    Args:
        data: DataFrame containing time series data
        fs: Sampling frequency
        chunk_size: Number of timepoints to process at once
        
    Returns:
        DataFrame containing PSD values for all channels
    """
    logger.info("Computing PSD for all channels...")
    logger.info(f"Input data shape: {data.shape}")
    
    # Split channels into groups for parallel processing
    n_channels = len(data.columns)
    n_jobs = min(os.cpu_count(), n_channels)
    channel_groups = np.array_split(data.columns, n_jobs)
    logger.info(f"Split {n_channels} channels into {n_jobs} groups for parallel processing")
    
    def process_group(channels: List[str]) -> pd.DataFrame:
        """Process a group of channels in chunks."""
        logger.info(f"Processing group with {len(channels)} channels")
        results = []
        total_chunks = (len(data) + chunk_size - 1) // chunk_size
        logger.info(f"Will process {total_chunks} chunks for this group")
        
        for i in range(0, len(data), chunk_size):
            chunk_num = i // chunk_size + 1
            logger.info(f"Processing chunk {chunk_num}/{total_chunks} for channels {channels[0]}-{channels[-1]}")
            chunk = data[channels].iloc[i:i + chunk_size]
            logger.info(f"Chunk shape: {chunk.shape}")
            freqs, psd_matrix = scipy.signal.welch(
                chunk, 
                fs, 
                window='hamming', 
                nperseg=fs*2, 
                noverlap=fs, 
                scaling='density', 
                axis=0
            )
            results.append(pd.DataFrame(psd_matrix, index=freqs, columns=channels))
            logger.info(f"PSD matrix shape: {psd_matrix.shape}")
        
        # Average PSD across chunks
        logger.info("Averaging PSD across chunks...")
        result = pd.concat(results).groupby(level=0).mean()
        logger.info(f"Final PSD shape for group: {result.shape}")
        return result
    
    # Process groups in parallel
    logger.info("Starting parallel processing of channel groups...")
    results = in_parallel(process_group, channel_groups, verbose=True)
    
    # Combine results
    logger.info("Combining results from all groups...")
    final_result = pd.concat(results, axis=1)
    logger.info(f"Final combined PSD shape: {final_result.shape}")
    return final_result

# def fooof_single_series(args: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
#     """Calculate FOOOF features for a single PSD.
#     
#     Args:
#         args: Tuple of (frequencies, PSD values)
#         
#     Returns:
#         Dictionary containing FOOOF features
#     """
#     # Import FOOOF here to avoid multiple deprecation warnings
#     from fooof import FOOOF
#     
#     freqs, psd_vals = args
#     try:
#         fm = FOOOF(
#             peak_width_limits=[1, 8],
#             max_n_peaks=6,
#             min_peak_height=0.1,
#             aperiodic_mode='knee'
#         )
#         fm.fit(freqs, psd_vals, (1, 40))
#         return {
#             'aperiodic_offset': fm.aperiodic_params_[0],
#             'aperiodic_exponent': fm.aperiodic_params_[1],
#             'r_squared': fm.r_squared_,
#             'error': fm.error_,
#             'num_peaks': fm.n_peaks_
#         }
#     except Exception as e:
#         logger.error(f"Error calculating FOOOF features: {str(e)}")
#         raise

def compute_entropy_single_series(series: np.ndarray) -> float:
    """Calculate Shannon entropy for a single time series.
    
    Args:
        series: Input time series
        
    Returns:
        Entropy value
    """
    try:
        # Use numpy's histogram with automatic bin selection
        hist, bins = np.histogram(series, bins='auto', density=True)
        probabilities = hist / np.sum(hist)
        probabilities = probabilities[probabilities > 0]  # Remove zero probabilities
        return -np.sum(probabilities * np.log2(probabilities))
    except Exception as e:
        logger.error(f"Error calculating entropy: {str(e)}")
        raise
