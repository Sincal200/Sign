import os

# Path to the file that needs fixing
file_path = r"C:\Users\sinca\PyCharmMiscProject\.venv\Lib\site-packages\tensorflowjs\write_weights.py"

# Read the file content
with open(file_path, 'r') as file:
    content = file.read()

# Replace the deprecated numpy types
content = content.replace('np.bool,', 'np.bool_,')
content = content.replace('np.bool]', 'np.bool_]')
content = content.replace('np.object]', 'np.object_]')

# Write the updated content back to the file
with open(file_path, 'w') as file:
    file.write(content)

print(f"Fixed numpy deprecated aliases in {file_path}")