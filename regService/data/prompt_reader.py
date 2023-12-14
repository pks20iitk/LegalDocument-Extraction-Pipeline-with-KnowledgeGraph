import os
import json

def get_lat_master_prompts():
    folder_path = 'regService/data/master_prompts'
    version_checker = {}
    # Iterate over the files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a regular file (not a subdirectory)
        if os.path.isfile(os.path.join(folder_path, filename)):
            version = filename.split('_V')[1].split('.')[0]
            version_checker[filename] = int(version)
    latest_filename = ''
    for keys,values in version_checker.items():
        if values==max(version_checker.values()):
            latest_filename = keys
    filepath = os.path.join(folder_path,latest_filename)
    with open(filepath,'r') as file:
        content = json.load(file)
        return content

def build_master_prompts(prompt_dict):
    folder_path = 'regService/data/master_prompts'
    version_checker = {}
    # Iterate over the files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a regular file (not a subdirectory)
        if os.path.isfile(os.path.join(folder_path, filename)):
            version = filename.split('_V')[1].split('.')[0]
            version_checker[filename] = int(version)
    latest_filename = ''
    for keys,values in version_checker.items():
        if values==max(version_checker.values()):
            latest_filename = keys
    filepath = os.path.join(folder_path,latest_filename)
    with open(filepath,'w') as file:
        json.dump(prompt_dict,file)
    return get_lat_master_prompts()

def get_prompts():
# Specify the folder path
    folder_path = 'regService/data/prompts'
    read_prompt_dict = {}
    # Iterate over the files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a regular file (not a subdirectory)
        if os.path.isfile(os.path.join(folder_path, filename)):
            # Create the full file path
            file_path = os.path.join(folder_path, filename)
            
            # Use 'with open' to open the file
            with open(file_path, 'r') as file:
                # Do operations on the file here
                content = json.load(file)
                
                # Example: Print the content of the file
                read_prompt_dict.update(content)
    return build_master_prompts(read_prompt_dict)
