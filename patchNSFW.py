import os

# Set this to the folder containing reactor_sfw.py
target_folder = "./ComfyUI/custom_nodes/ComfyUI-ReActor/scripts"
target_file = "reactor_sfw.py"
full_path = os.path.join(target_folder, target_file)

# Patched function code
patched_function = '''def nsfw_image(img_data, model_path: str):
    return False
'''

def patch_nsfw_function(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find the start of the nsfw_image function
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("def nsfw_image("):
            start_idx = i
            break

    if start_idx is None:
        print("Function nsfw_image not found.")
        return

    # Find end of function (next def/class or end of file)
    end_idx = start_idx + 1
    while end_idx < len(lines) and not lines[end_idx].strip().startswith(("def ", "class ")):
        end_idx += 1

    # Replace old function with patched one
    lines[start_idx:end_idx] = [patched_function + '\n']

    # Save patched file
    with open(file_path, 'w') as f:
        f.writelines(lines)

    print(f"Patched 'nsfw_image' in {file_path}")

# Run the patch
patch_nsfw_function(full_path)
