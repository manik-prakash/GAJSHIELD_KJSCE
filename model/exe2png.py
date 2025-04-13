import math
from PIL import Image
import numpy as np

def exe_to_image(exe_path, output_path):
    # Read the executable file in binary mode
    with open(exe_path, 'rb') as f:
        byte_data = f.read()
    
    # Calculate dimensions for a square-ish image
    data_len = len(byte_data)
    width = math.ceil(math.sqrt(data_len))
    height = width
    
    # Create a numpy array and pad with zeros if needed
    img_array = np.zeros((height, width), dtype=np.uint8)
    np_data = np.frombuffer(byte_data, dtype=np.uint8)
    img_array.flat[:len(np_data)] = np_data
    
    # Create and save the image
    img = Image.fromarray(img_array, mode='L')
    img.save(output_path)
    print(f"Image saved to {output_path} (Dimensions: {width}x{height})")

def main():
    # Define input and output paths
    exe_path = 'JaffaCakes118_b23c693ab0321b30bf3272efb39ef280.exe'
    output_path = 'test_bytecode_image.png'

    # Call the function to convert the executable to an image
    exe_to_image(exe_path, output_path)

# Ensure the script runs only when executed directly
if __name__ == "__main__":
    main()