def dei_print(x):
    print(f'\n\n--\n{x}\n--\n\n')

def dei_save(path, file):
    import os
    import torch
    
    os.makedirs(os.path.expanduser(f'~/data'), exist_ok=True)
    path = f'~/data/{path}.pt'
    path = os.path.expanduser(path)
    try:
        print(f'Saving tensor to {path}')
        torch.save(file, path)
    except Exception as e:
        print(f"Error saving tensor: {e}")

def dei_load(path):
    import os
    import torch
    
    path = f'~/data/{path}.pt'
    path = os.path.expanduser(path)
    try:
        tensor = torch.load(path)
        return tensor
    except Exception as e:
        print(f"Error loading tensor: {e}")
        return None