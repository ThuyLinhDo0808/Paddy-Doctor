import yaml
import os
import warnings
import sys
import torch

def read_config(config_path):

    """
    Load a YAML configuration file safely and handle errors.
    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration as a dictionary.
    
    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file is empty or not valid YAML.
        PermissionError: If there are permission issues accessing the file.
        Exception: For any other unexpected errors.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[ERROR] Config file not found: {config_path}")
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if config is None:
            raise ValueError(f"[ERROR] Config file is empty or not valid YAML: {config_path}")
        return  config
    
    except yaml.YAMLError as e:
        raise ValueError(f"[ERROR] YAML syntax error in config file: {e}")
    
    except PermissionError as e:
        raise PermissionError(f"[ERROR] Permission denied when accessing config file: {config_path}")
    
    except Exception as e:
        warnings.warn(f"[WARN] Unexpected error reading config: {e}")
        sys.exit(1)

# Force CPU fallback if CUDA is not available
def get_device(force_cpu=False):

    if force_cpu or not torch.cuda.is_available():
        print("[INFO] Using CPU")
        return torch.device("cpu")
    
    try:
        name = torch.cuda.get_device_name(0)
        print(f"[INFO] Using GPU: {name}")
        return torch.device("cuda")
    except Exception:
        print("[WARN] CUDA available but failed to access device. Falling back to CPU.")
        return torch.device("cpu")
    
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        """
        Initialize the EarlyStopping object.

        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            delta (float): Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        """
        Call method to check if early stopping criteria are met.

        Args:
            score (float): Current score to compare with the best score.
        """
        score = -val_score

        if self.best_score is None:
            self.best_score = score
            return False
        
        if score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.counter = 0
        
        return False
            