"""
Configuration management for the gender and age prediction project.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the application."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.config_path = config_path or "config/config.yaml"
        self.config = self._load_default_config()
        
        if Path(self.config_path).exists():
            self._load_config()
        else:
            logger.info(f"Config file not found at {self.config_path}, using defaults")
            self._save_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "model": {
                "name": "modern",
                "device": "auto",  # auto, cpu, cuda
                "batch_size": 1,
                "confidence_threshold": 0.5
            },
            "face_detection": {
                "cascade_path": None,
                "scale_factor": 1.1,
                "min_neighbors": 5,
                "min_size": [30, 30]
            },
            "data": {
                "input_dir": "data/input",
                "output_dir": "data/output",
                "synthetic_dir": "data/synthetic",
                "max_image_size": [224, 224]
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/app.log"
            },
            "web_app": {
                "host": "localhost",
                "port": 8501,
                "title": "Gender & Age Prediction",
                "debug": False
            },
            "age_buckets": ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"],
            "gender_labels": ["Male", "Female"]
        }
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    loaded_config = yaml.safe_load(f)
                elif self.config_path.endswith('.json'):
                    loaded_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {self.config_path}")
                
                # Merge with defaults
                self._merge_config(self.config, loaded_config)
                logger.info(f"Configuration loaded from {self.config_path}")
                
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            logger.info("Using default configuration")
    
    def _merge_config(self, default: Dict[str, Any], loaded: Dict[str, Any]):
        """Recursively merge loaded config with defaults."""
        for key, value in loaded.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
    
    def _save_config(self):
        """Save current configuration to file."""
        try:
            config_dir = Path(self.config_path).parent
            config_dir.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                elif self.config_path.endswith('.json'):
                    json.dump(self.config, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config to {self.config_path}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation like 'model.device')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]):
        """
        Update multiple configuration values.
        
        Args:
            updates: Dictionary of key-value pairs to update
        """
        for key, value in updates.items():
            self.set(key, value)
    
    def save(self):
        """Save current configuration to file."""
        self._save_config()
    
    def reload(self):
        """Reload configuration from file."""
        if Path(self.config_path).exists():
            self._load_config()
        else:
            logger.warning(f"Config file {self.config_path} does not exist")


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config
