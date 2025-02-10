import os

# Folder path
folder_path = "./data/LS 2.13.23"

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a JPG and starts with 'DJI_'
    if filename.startswith("DJI_") and filename.endswith(".JPG"):
        # print(f"Found: {filename}")
        # New filename with prefix
        new_filename = f"LS21323_{filename}"

        # Full paths
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_filename)

        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed: {filename} -> {new_filename}")