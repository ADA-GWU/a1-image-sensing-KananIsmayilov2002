import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Stereo Vision: Census Transform & Block Matching')
    
    parser.add_argument('--dataset', type=str, default='dataset3', 
                        help='Name of the dataset folder (e.g., dataset1, dataset2)')
    parser.add_argument('--window_size', type=int, default=11, 
                        help='Size of the matching window')
    parser.add_argument('--max_disparity', type=int, default=64,
                        help='Maximum search range for disparity')
    
    return parser.parse_args()

def fetch_stereo_pair(dataset_name):
    """Loads the left and right grayscale images from the specified dataset folder."""
    folder_path = os.path.join('images', 'original', dataset_name)
    
    
    left_img = os.path.join(folder_path, 'left.jpeg')
    right_img = os.path.join(folder_path, 'right.jpeg')
    

    print(f"Loading images from '{dataset_name}'...")
    imgL = cv2.imread(left_img, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(right_img, cv2.IMREAD_GRAYSCALE)

    return imgL, imgR

def apply_census_transform(img, win_size):
    """Encodes local pixel structures into a 64-bit integer map."""
    h, w = img.shape
    pad = win_size // 2
    census_map = np.zeros((h, w), dtype=np.uint64)

    center_region = img[pad : h - pad, pad : w - pad]

    bit_shift = 0
    for row_offset in range(win_size):
        for col_offset in range(win_size):
            # Skip the center pixel itself
            if row_offset == pad and col_offset == pad:
                continue

            neighbor_region = img[row_offset : h - win_size + row_offset + 1, 
                                  col_offset : w - win_size + col_offset + 1]
            
            # Compare and shift bits
            is_smaller = (neighbor_region < center_region).astype(np.uint64)
            census_map[pad : h - pad, pad : w - pad] |= (is_smaller << bit_shift)
            bit_shift += 1

    return census_map

def compute_disparity(left_img, right_img, win_size, max_disp):
    """Calculates disparity by matching Census Transform bit strings."""
    print(f"Processing Disparity (Window: {win_size}, Max Disp: {max_disp})")

    # Downscale images by 50% to speed up processing
    new_w = int(left_img.shape[1] * 0.5)
    new_h = int(left_img.shape[0] * 0.5)
    
    left_small = cv2.resize(left_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    right_small = cv2.resize(right_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    print("Applying Census Transform to both images...")
    census_L = apply_census_transform(left_small, win_size)
    census_R = apply_census_transform(right_small, win_size)

    h, w = left_small.shape
    disp_map = np.zeros((h, w), dtype=np.uint8)
    pad = win_size // 2

    print("Running Block Matching...")
    for y in range(pad, h - pad):
        if y % 50 == 0:
            print(f"  Evaluating row {y} of {h}...")
            
        for x in range(max_disp + pad, w - pad):
            target_val = census_L[y, x]
            
            best_match_disp = 0
            lowest_cost = float('inf')

            # Search along the epipolar line (horizontal)
            for d in range(max_disp):
                candidate_val = census_R[y, x - d]
                
                # Calculate Hamming distance (number of differing bits)
                hamming_dist = bin(target_val ^ candidate_val).count('1')

                if hamming_dist < lowest_cost:
                    lowest_cost = hamming_dist
                    best_match_disp = d

            disp_map[y, x] = best_match_disp

    
    normalized_disp = cv2.normalize(disp_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return normalized_disp, left_small, right_small

def display_and_save(img_L, img_R, disp_map, save_destination):
    """Creates a side-by-side plot of the inputs and the resulting map."""
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img_L, cmap='gray')
    plt.title('Left Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img_R, cmap='gray')
    plt.title('Right Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(disp_map, cmap='inferno')
    plt.title('Disparity Map')
    plt.colorbar(fraction=0.046, pad=0.04, label='Pixels')
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(save_destination, dpi=150, bbox_inches='tight')
    print(f"--> Saved visualization to: {save_destination}")

    plt.show()

if __name__ == "__main__":
    args = parse_arguments()

    out_dir = os.path.join('images', 'output', args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    
    left_raw, right_raw = fetch_stereo_pair(args.dataset)

    
    result_map, left_scaled, right_scaled = compute_disparity(
        left_raw, right_raw, args.window_size, args.max_disparity
    )

    
    viz_filename = f"Window_size_{args.window_size}_{args.dataset}.png"
    viz_filepath = os.path.join(out_dir, viz_filename)
    
    display_and_save(left_scaled, right_scaled, result_map, viz_filepath)

