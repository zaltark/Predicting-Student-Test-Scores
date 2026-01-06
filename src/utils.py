import yaml
import logging
import os

def load_config(config_path="config.yaml"):
    """Loads the configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def setup_logger(name, log_file, level=logging.INFO):
    """Sets up a logger that writes to console and file."""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    
    return logger
