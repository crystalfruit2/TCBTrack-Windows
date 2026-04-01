import os
import shutil

train_dir = r'datasets/mot/train'
val_dir = r'datasets/mot/val'

# Create val directory if it doesn't exist
os.makedirs(val_dir, exist_ok=True)

# Find all sequence folders in train (e.g., MOT17-02-DPM)
folders = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]

for folder in folders:
    print(f"Processing {folder}...")
    src_folder = os.path.join(train_dir, folder)
    dst_folder = os.path.join(val_dir, folder)
    
    # Create the target img1 directory
    os.makedirs(os.path.join(dst_folder, 'img1'), exist_ok=True)
    
    # Copy Ground Truth (gt) and det folders
    for sub in ['gt', 'det']:
        if os.path.exists(os.path.join(src_folder, sub)):
            shutil.copytree(os.path.join(src_folder, sub), os.path.join(dst_folder, sub), dirs_exist_ok=True)
            
    # Read the original seqinfo.ini
    seqinfo_path = os.path.join(src_folder, 'seqinfo.ini')
    with open(seqinfo_path, 'r') as f:
        lines = f.readlines()
        
    # Find the sequence length
    seq_length = 0
    for line in lines:
        if 'seqLength' in line:
            seq_length = int(line.strip().split('=')[1])
            break
            
    half = seq_length // 2
    
    # Copy ONLY the second half of the images
    img_src_dir = os.path.join(src_folder, 'img1')
    img_dst_dir = os.path.join(dst_folder, 'img1')
    
    for i in range(half + 1, seq_length + 1):
        img_name = f"{i:06d}.jpg"
        src_img = os.path.join(img_src_dir, img_name)
        dst_img = os.path.join(img_dst_dir, img_name)
        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)
            
    # Write the fixed seqinfo.ini with the halved length
    with open(os.path.join(dst_folder, 'seqinfo.ini'), 'w') as f:
        for line in lines:
            if 'seqLength' in line:
                f.write(f"seqLength={seq_length - half}\n")
            else:
                f.write(line)

print("\nSuccess! All folders perfectly halved and prepped for Windows.")