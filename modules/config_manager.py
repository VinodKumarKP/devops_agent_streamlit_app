import os
from typing import Dict

import streamlit as st
import yaml


class ConfigManager:
    """
    Manages application configuration loading and access.
    """

    def __init__(self):
        """Initialize configuration manager."""
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """
        Load configuration from YAML file.

        Returns:
            Dict: Configuration dictionary
        """
        directory_name = os.path.dirname(__file__)
        config_path = os.path.join(os.path.dirname(directory_name), 'config', 'sidebar.yaml')

        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            st.error(f"Configuration file not found: {config_path}")
            return {}
        except yaml.YAMLError as e:
            st.error(f"Error parsing configuration file: {e}")
            return {}
