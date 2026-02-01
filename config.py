import os
import yaml
import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass
class TimeOfDayConfig:
    """Configuration for a specific time of day."""
    sun_altitude: float = 45.0
    sun_azimuth: float = 180.0
    temperature_factor: float = 0.5  # 0=cold, 1=hot

@dataclass
class QuantumConfig:
    """Configuration for quantum computing aspects."""
    use_quantum: bool = True
    use_real_device: bool = True
    api_token: Optional[str] = None
    optimization_level: int = 3

@dataclass
class DataConfig:
    """Configuration for data paths."""
    data_path: str = None
    result_path: str = None
    use_observed_data: bool = True

@dataclass
class ModelConfig:
    """Configuration for the urban model."""
    urbs_coeffs: List[float] = field(default_factory=lambda: [0.125] * 8)
    civitas_coeffs: List[float] = field(default_factory=lambda: [0.5, 0.5])
    learn_coeffs_from_data: bool = False
    use_quantum_coeffs: bool = True  # Explicitly including this attribute

@dataclass
class Config:
    """Main configuration for the quantum urban planning model."""
    model: ModelConfig = field(default_factory=ModelConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    data: DataConfig = field(default_factory=DataConfig)
    times_of_day: Dict[str, TimeOfDayConfig] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize with default values if not provided."""
        # Initialize default paths if not provided
        if self.data.data_path is None:
            self.data.data_path = os.path.join(os.getcwd(), 'data')
        
        if self.data.result_path is None:
            self.data.result_path = os.path.join(os.getcwd(), 'results')
            
        # Initialize default times of day if not provided
        if not self.times_of_day:
            self.times_of_day = {
                'morning': TimeOfDayConfig(sun_altitude=30.0, sun_azimuth=90.0, temperature_factor=0.3),
                'afternoon': TimeOfDayConfig(sun_altitude=60.0, sun_azimuth=180.0, temperature_factor=0.8),
                'evening': TimeOfDayConfig(sun_altitude=20.0, sun_azimuth=270.0, temperature_factor=0.5),
                'night': TimeOfDayConfig(sun_altitude=0.0, sun_azimuth=0.0, temperature_factor=0.2)
            }
    
    @classmethod
    def from_yaml(cls, filepath):
        """Load configuration from a YAML file."""
        try:
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
                
            # Create main config object
            config = cls()
            
            # Update model config
            if 'model' in config_dict:
                model_dict = config_dict['model']
                if 'urbs_coeffs' in model_dict:
                    config.model.urbs_coeffs = model_dict['urbs_coeffs']
                if 'civitas_coeffs' in model_dict:
                    config.model.civitas_coeffs = model_dict['civitas_coeffs']
                if 'learn_coeffs_from_data' in model_dict:
                    config.model.learn_coeffs_from_data = model_dict['learn_coeffs_from_data']
                if 'use_quantum_coeffs' in model_dict:
                    config.model.use_quantum_coeffs = model_dict['use_quantum_coeffs']
                    
            # Update quantum config
            if 'quantum' in config_dict:
                quantum_dict = config_dict['quantum']
                if 'use_quantum' in quantum_dict:
                    config.quantum.use_quantum = quantum_dict['use_quantum']
                if 'use_real_device' in quantum_dict:
                    config.quantum.use_real_device = quantum_dict['use_real_device']
                if 'api_token' in quantum_dict:
                    config.quantum.api_token = quantum_dict['api_token']
                if 'optimization_level' in quantum_dict:
                    config.quantum.optimization_level = quantum_dict['optimization_level']
                    
            # Update data config
            if 'data' in config_dict:
                data_dict = config_dict['data']
                if 'data_path' in data_dict:
                    config.data.data_path = data_dict['data_path']
                if 'result_path' in data_dict:
                    config.data.result_path = data_dict['result_path']
                if 'use_observed_data' in data_dict:
                    config.data.use_observed_data = data_dict['use_observed_data']
                    
            # Update times of day
            if 'times_of_day' in config_dict:
                times_dict = config_dict['times_of_day']
                config.times_of_day = {}
                
                for time_name, time_params in times_dict.items():
                    config.times_of_day[time_name] = TimeOfDayConfig(
                        sun_altitude=time_params.get('sun_altitude', 45.0),
                        sun_azimuth=time_params.get('sun_azimuth', 180.0),
                        temperature_factor=time_params.get('temperature_factor', 0.5)
                    )
                    
            return config
            
        except Exception as e:
            warnings.warn(f"Config file {filepath} not found or invalid, using defaults: {str(e)}")
            return cls()