import os
import json
import yaml

def get_model_config():
    """
    Get model configuration from config file or environment variables,
    with reasonable defaults if neither is available.
    
    Returns:
        dict: Model configuration parameters
    """
    # Get the home directory path
    home_dir = os.path.expanduser("~")
    
    config = {
        "v_high": 105.0,
        "v_low": 95.0,
        "p": 0.5,
        "alpha": 0.2,
        "data_path": os.path.join(home_dir, "code/Algorithmic_Trading_and_HFT_Research/HFT_Binance_data_fetcher/crypto_data"),
        "symbol": "btcusdt",
        "update_frequency": 1.0
    }
    
    # Try to load from config file
    config_file = os.path.join(os.path.dirname(__file__), '../../config.yaml')
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    config.update(file_config)
        except Exception as e:
            print(f"Error loading config file: {str(e)}")
    
    # Override with environment variables if they exist
    env_vars = {
        "GM_V_HIGH": "v_high",
        "GM_V_LOW": "v_low",
        "GM_P": "p",
        "GM_ALPHA": "alpha",
        "GM_DATA_PATH": "data_path",
        "GM_SYMBOL": "symbol",
        "GM_UPDATE_FREQUENCY": "update_frequency"
    }
    
    for env_var, config_key in env_vars.items():
        if env_var in os.environ:
            try:
                # Handle type conversion based on the default type
                val = os.environ[env_var]
                if isinstance(config[config_key], float):
                    config[config_key] = float(val)
                elif isinstance(config[config_key], int):
                    config[config_key] = int(val)
                else:
                    config[config_key] = val
            except Exception as e:
                print(f"Error processing environment variable {env_var}: {str(e)}")
    
    # Ensure data directory exists
    os.makedirs(config["data_path"], exist_ok=True)
    
    return config