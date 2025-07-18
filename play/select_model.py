import os

def list_available_models(model_folder="./models"):
    """
    List all available DQN models in the specified folder
    
    Args:
        model_folder (str): Path to the folder containing models
    
    Returns:
        list: List of model paths (without .zip extension)
    """
    model_files = []
    
    # Look for .zip files that match the DQN model pattern
    for file in os.listdir(model_folder):
        if file.endswith('.zip'):
            model_path = os.path.join(model_folder, file[:-4])  # Remove .zip extension
            model_files.append(model_path)
    
    return sorted(model_files)

def select_model_interactive():
    """
    Interactive model selection function
    
    Returns:
        str: Selected model path or None if cancelled
    """
    print("\n" + "="*50)
    print("AVAILABLE TRAINED MODELS")
    print("="*50)
    
    models = list_available_models()
    
    if not models:
        print("No trained models found!")
        return None
    
    for i, model_path in enumerate(models, 1):
        model_name = os.path.basename(model_path)
        print(f"{i:2d}. {model_name}")
    
    while True:
        try:
            choice = input("\nEnter model number (0 to cancel): ").strip()
            if choice == '0':
                print("Selection cancelled.")
                return None
                
            selection = int(choice)
            if 1 <= selection <= len(models):
                return models[selection - 1]
            print(f"Please enter 1-{len(models)} or 0 to cancel")
                
        except ValueError:
            print("Please enter a number")
        except KeyboardInterrupt:
            print("\nSelection cancelled.")
            return None