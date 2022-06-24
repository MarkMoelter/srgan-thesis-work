def create_pth(x: str):
    """Create a folder if it does not already exist at that path"""
    import os
    if not os.path.exists(x):
        os.makedirs(x)
