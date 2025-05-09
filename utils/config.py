import yaml

def load_config(path="config/config.yaml") -> dict:
    """
    Loads a YAML config file and returns it as a Python dictionary.
    
    Args:
        path (str): Path to the YAML config file.
    
    Returns:
        dict: Parsed config.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)
