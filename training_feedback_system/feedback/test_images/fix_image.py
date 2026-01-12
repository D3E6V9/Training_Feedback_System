import base64
import os

def fix_image():
    # Read the base64 data file
    image_path = 'feedback_form.jpg'
    with open(image_path, 'r') as f:
        data = f.read()
    
    # Clean up the data
    if data.startswith('data:image'):
        # Remove the data URL prefix
        data = data.split(',', 1)[1]
    
    # Ensure proper padding
    missing_padding = len(data) % 4
    if missing_padding:
        data += '=' * (4 - missing_padding)
    
    # Convert to binary and save back
    try:
        binary_data = base64.b64decode(data)
        temp_path = 'temp_form.jpg'
        with open(temp_path, 'wb') as f:
            f.write(binary_data)
        
        # Replace original file with the fixed version
        os.replace(temp_path, image_path)
        print("Successfully converted to image file")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    fix_image()
