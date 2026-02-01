import numpy as np
import os
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
from urban_potential_field import UrbanPotentialField
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeotiffProcessor:
    def __init__(self, output_dir="./output"):
        """
        Initialize GeotiffProcessor to handle GeoTIFF inputs for urban potential field analysis.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save outputs
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Dictionary to store raster data
        self.rasters = {}
        self.common_meta = None
        self.grid_shape = None
    
    def load_geotiff(self, factor_name, filepath):
        """
        Load a GeoTIFF file and store it in the rasters dictionary.
        
        Parameters:
        -----------
        factor_name : str
            Name of the factor (e.g., 'building_density')
        filepath : str
            Path to the GeoTIFF file
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            with rasterio.open(filepath) as src:
                logger.info(f"Loading {factor_name} from {filepath}")
                # Read the first band (index 1)
                data = src.read(1, masked=True)
                
                # Store raster data and metadata
                self.rasters[factor_name] = {
                    'data': data,
                    'meta': src.meta.copy(),
                    'filepath': filepath
                }
                
                # If first raster loaded, use its metadata as base
                if self.common_meta is None:
                    self.common_meta = src.meta.copy()
                    self.grid_shape = data.shape
                
                return True
        except RasterioIOError as e:
            logger.error(f"Error opening {filepath}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error processing {filepath}: {str(e)}")
            return False
    
    def align_rasters(self):
        """
        Align all loaded rasters to have the same dimensions and spatial reference.
        Uses the first loaded raster as the reference.
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if not self.rasters or self.common_meta is None:
            logger.error("No rasters loaded or common metadata not set")
            return False
        
        reference_meta = self.common_meta
        
        for factor_name, raster_info in self.rasters.items():
            # Skip if this is the reference raster or already aligned
            if raster_info['meta'] == reference_meta:
                continue
            
            logger.info(f"Aligning {factor_name} to reference grid")
            
            # Read source data
            with rasterio.open(raster_info['filepath']) as src:
                source_data = src.read(1)
                
                # Create destination array
                dest_data = np.zeros(self.grid_shape, dtype=raster_info['meta']['dtype'])
                
                # Reproject to match reference
                reproject(
                    source=source_data,
                    destination=dest_data,
                    src_transform=raster_info['meta']['transform'],
                    src_crs=raster_info['meta']['crs'],
                    dst_transform=reference_meta['transform'],
                    dst_crs=reference_meta['crs'],
                    resampling=Resampling.bilinear
                )
                
                # Update the raster data
                self.rasters[factor_name]['data'] = dest_data
                self.rasters[factor_name]['meta'] = reference_meta.copy()
        
        logger.info("All rasters aligned to reference grid")
        return True
    
    def process_potential_field(self, alpha=0.5, beta=0.5, temperature=0.1):
        """
        Process all rasters to generate the urban potential field and probability map.
        
        Parameters:
        -----------
        alpha : float
            Weight for V urbs
        beta : float
            Weight for V civitas
        temperature : float
            Temperature parameter in the Boltzmann distribution
        
        Returns:
        --------
        tuple
            (UrbanPotentialField object, success flag)
        """
        # Check if we have all required factors
        v_urbs_factors = [
            'building_density', 'building_height', 'tree_height',
            'wall_constraint', 'mean_depth', 'street_centrality'
        ]
        
        v_civitas_factors = [
            'poi_density', 'pedestrian_accessibility', 
            'isovist', 'vehicle_density'
        ]
        
        # Check if all required factors are loaded
        missing_factors = []
        for factor in v_urbs_factors + v_civitas_factors:
            if factor not in self.rasters:
                missing_factors.append(factor)
        
        if missing_factors:
            logger.warning(f"Missing factors: {', '.join(missing_factors)}")
            logger.warning("Using zero arrays for missing factors")
        
        # Create Urban Potential Field object
        grid_size = self.grid_shape
        upf = UrbanPotentialField(grid_size)
        
        # Prepare V urbs factors
        v_urbs_data = {}
        for factor in v_urbs_factors:
            if factor in self.rasters:
                v_urbs_data[factor] = self.rasters[factor]['data']
                # Fill masked values (if any) with zeros
                if isinstance(v_urbs_data[factor], np.ma.MaskedArray):
                    v_urbs_data[factor] = v_urbs_data[factor].filled(0)
            else:
                v_urbs_data[factor] = np.zeros(grid_size)
        
        # Prepare V civitas factors
        v_civitas_data = {}
        for factor in v_civitas_factors:
            if factor in self.rasters:
                v_civitas_data[factor] = self.rasters[factor]['data']
                # Fill masked values (if any) with zeros
                if isinstance(v_civitas_data[factor], np.ma.MaskedArray):
                    v_civitas_data[factor] = v_civitas_data[factor].filled(0)
            else:
                v_civitas_data[factor] = np.zeros(grid_size)
        
        # Set V urbs factors
        upf.set_v_urbs_factors(
            v_urbs_data['building_density'],
            v_urbs_data['building_height'],
            v_urbs_data['tree_height'],
            v_urbs_data['wall_constraint'],
            v_urbs_data['mean_depth'],
            v_urbs_data['street_centrality']
        )
        
        # Set V civitas factors
        upf.set_v_civitas_factors(
            v_civitas_data['poi_density'],
            v_civitas_data['pedestrian_accessibility'],
            v_civitas_data['isovist'],
            v_civitas_data['vehicle_density']
        )
        
        # Generate potential field
        upf.generate_potential_field(alpha=alpha, beta=beta)
        
        # Generate probability distribution
        upf.generate_probability_distribution(temperature=temperature)
        
        # Set ground truth if available
        if 'ground_truth' in self.rasters:
            ground_truth_data = self.rasters['ground_truth']['data']
            if isinstance(ground_truth_data, np.ma.MaskedArray):
                ground_truth_data = ground_truth_data.filled(0)
            upf.set_ground_truth(ground_truth_data)
        
        # Generate report
        upf.generate_summary_report(self.output_dir)
        
        # Save results as GeoTIFFs
        self.save_results_as_geotiff(upf)
        
        return upf, True
    
    def save_results_as_geotiff(self, upf):
        """
        Save the potential field and probability distribution as GeoTIFF files.
        
        Parameters:
        -----------
        upf : UrbanPotentialField
            The urban potential field object
        """
        if self.common_meta is None:
            logger.error("Cannot save results: reference metadata not available")
            return False
        
        meta = self.common_meta.copy()
        
        # Update metadata for output
        meta.update(
            dtype=rasterio.float32,
            count=1,
            compress='lzw',
            nodata=np.nan
        )
        
        # Save potential field
        potential_field_path = os.path.join(self.output_dir, "potential_field.tif")
        with rasterio.open(potential_field_path, 'w', **meta) as dst:
            dst.write(upf.potential_field.astype(rasterio.float32), 1)
        logger.info(f"Potential field saved to {potential_field_path}")
        
        # Save probability distribution
        probability_path = os.path.join(self.output_dir, "probability_distribution.tif")
        with rasterio.open(probability_path, 'w', **meta) as dst:
            dst.write(upf.probability_dist.astype(rasterio.float32), 1)
        logger.info(f"Probability distribution saved to {probability_path}")
        
        # Save V urbs
        v_urbs_path = os.path.join(self.output_dir, "v_urbs.tif")
        with rasterio.open(v_urbs_path, 'w', **meta) as dst:
            dst.write(upf.v_urbs.astype(rasterio.float32), 1)
        logger.info(f"V urbs saved to {v_urbs_path}")
        
        # Save V civitas
        v_civitas_path = os.path.join(self.output_dir, "v_civitas.tif")
        with rasterio.open(v_civitas_path, 'w', **meta) as dst:
            dst.write(upf.v_civitas.astype(rasterio.float32), 1)
        logger.info(f"V civitas saved to {v_civitas_path}")
        
        return True