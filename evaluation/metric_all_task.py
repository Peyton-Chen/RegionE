import os
import cv2
import torch
import lpips
import argparse
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torchvision.transforms as transforms

def calculate_image_metrics(folder1_path, folder2_path):
    """
    Calculate PSNR, SSIM and LPIPS metrics for images with the same names in two folders
    
    Args:
        folder1_path (str): Path to the first folder (usually reference images)
        folder2_path (str): Path to the second folder (usually images to be evaluated)
    
    Returns:
        dict: Dictionary containing metrics for each image and average metrics
    """
    
    # Check if folders exist
    if not os.path.exists(folder1_path) or not os.path.exists(folder2_path):
        raise ValueError("Specified folder path does not exist")
    
    # Get all image files in folder 1
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    folder1_files = {f for f in os.listdir(folder1_path) 
                    if os.path.splitext(f.lower())[1] in valid_extensions}
    
    # Get all image files in folder 2
    folder2_files = {f for f in os.listdir(folder2_path) 
                    if os.path.splitext(f.lower())[1] in valid_extensions}
    
    # Find files with the same names in both folders
    common_files = folder1_files.intersection(folder2_files)
    
    if not common_files:
        raise ValueError("No images with matching names found in both folders")
    
    print(f"Found {len(common_files)} images with matching names")
    
    # Initialize LPIPS model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Store results
    results = {
        'individual_metrics': {},
        'average_metrics': {}
    }
    
    psnr_values = []
    ssim_values = []
    lpips_values = []
    
    print("Starting metric calculation...")
    
    for i, filename in enumerate(sorted(common_files)):
        print(f"Processing image {i+1}/{len(common_files)}: {filename}")
        
        # Read images
        img1_path = os.path.join(folder1_path, filename)
        img2_path = os.path.join(folder2_path, filename)
        
        try:
            # Read images using PIL
            img1_pil = Image.open(img1_path).convert('RGB')
            img2_pil = Image.open(img2_path).convert('RGB')
            
            # Ensure both images have the same dimensions
            if img1_pil.size != img2_pil.size:
                print(f"Warning: Size mismatch for {filename}, will resize to match")
                # Resize img2 to match img1's dimensions
                img2_pil = img2_pil.resize(img1_pil.size, Image.LANCZOS)
            
            # Convert to numpy arrays for PSNR and SSIM calculation
            img1_np = np.array(img1_pil)
            img2_np = np.array(img2_pil)
            
            # Calculate PSNR
            psnr_value = psnr(img1_np, img2_np)
            
            # Calculate SSIM
            ssim_value = ssim(img1_np, img2_np, multichannel=True, channel_axis=-1)
            
            # Calculate LPIPS
            img1_tensor = transform(img1_pil).unsqueeze(0).to(device)
            img2_tensor = transform(img2_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                lpips_value = lpips_model(img1_tensor, img2_tensor).item()
            
            # Store results for individual image
            results['individual_metrics'][filename] = {
                'PSNR': psnr_value,
                'SSIM': ssim_value,
                'LPIPS': lpips_value
            }
            
            # Add to lists for calculating averages
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
            lpips_values.append(lpips_value)
            
        except Exception as e:
            print(f"Error processing image {filename}: {str(e)}")
            continue
    
    # Calculate average metrics
    if psnr_values:
        results['average_metrics'] = {
            'PSNR': np.mean(psnr_values),
            'SSIM': np.mean(ssim_values),
            'LPIPS': np.mean(lpips_values)
        }
        
        # Print results
        print("\n" + "="*50)
        print("Calculation complete!")
        print("="*50)
        print(f"Average PSNR: {results['average_metrics']['PSNR']:.4f}")
        print(f"Average SSIM: {results['average_metrics']['SSIM']:.4f}")
        print(f"Average LPIPS: {results['average_metrics']['LPIPS']:.4f}")
        print("="*50)
        
        # Display detailed results for the first 5 images
        print("\nDetailed results for the first 5 images:")
        for i, (filename, metrics) in enumerate(list(results['individual_metrics'].items())[:5]):
            print(f"{filename}:")
            print(f"  PSNR: {metrics['PSNR']:.4f}")
            print(f"  SSIM: {metrics['SSIM']:.4f}")
            print(f"  LPIPS: {metrics['LPIPS']:.4f}")
    
    return results

def save_results_to_csv(results, output_path='image_metrics_results.csv'):
    """
    Save results to a CSV file
    
    Args:
        results (dict): Return value from calculate_image_metrics function
        output_path (str): Output CSV file path
    """
    import pandas as pd
    
    # Create DataFrame
    data = []
    for filename, metrics in results['individual_metrics'].items():
        data.append({
            'Filename': filename,
            'PSNR': metrics['PSNR'],
            'SSIM': metrics['SSIM'],
            'LPIPS': metrics['LPIPS']
        })
    
    df = pd.DataFrame(data)
    
    # Add average row
    avg_row = {
        'Filename': 'AVERAGE',
        'PSNR': results['average_metrics']['PSNR'],
        'SSIM': results['average_metrics']['SSIM'],
        'LPIPS': results['average_metrics']['LPIPS']
    }
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


# Usage example
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder1", type=str, required=True)
    parser.add_argument("--folder2", type=str, required=True)
    args = parser.parse_args()

    for task in os.listdir(args.folder1):
        folder1 = os.path.join(args.folder1, task, 'generation')
        folder2 = os.path.join(args.folder2, task, 'generation')
        output = os.path.join(args.folder2, task, 'metric.csv')

        try:
            # Calculate metrics
            results = calculate_image_metrics(folder1, folder2)
            
            # Save results to CSV file
            save_results_to_csv(results, output_path=output)
            
        except Exception as e:
            print(f"Execution error: {str(e)}")