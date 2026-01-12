import base64
import os
import sys

def save_image(file_path):
    # First read the existing file that contains the base64 data
    with open(file_path, 'r') as f:
        base64_data = f.read()

    # Remove the data URL prefix if present
    if base64_data.startswith('data:image/'):
        base64_data = base64_data.split(',', 1)[1]

    # Convert base64 to binary
    binary_data = base64.b64decode(base64_data)

    # Write binary data to file
    with open(file_path, 'wb') as f:
        f.write(binary_data)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        save_image(sys.argv[1])
    else:
        print("Please provide the output file path")
