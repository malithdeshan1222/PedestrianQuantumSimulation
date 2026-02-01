import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
import seaborn as sns
from scipy.ndimage import zoom

def ensure_same_shape(array1, array2):
    """
    Resize arrays to match shapes via interpolation if needed.
    
    Parameters:
        array1, array2: Arrays to match in shape
        
    Returns:
        tuple: (array1, array2) with matching shapes
    """
    if array1.shape == array2.shape:
        return array1, array2
        
    print(f"Reshaping arrays to match: {array1.shape} -> {array2.shape}")
    
    # Determine target shape (use the larger one by default)
    if np.prod(array1.shape) > np.prod(array2.shape):
        target_shape = array1.shape
        # Upscale array2 to match array1
        zoom_factors = np.array(target_shape) / np.array(array2.shape)
        array2_resized = zoom(array2, zoom_factors, order=1)
        return array1, array2_resized
    else:
        target_shape = array2.shape
        # Upscale array1 to match array2
        zoom_factors = np.array(target_shape) / np.array(array2.shape)
        array1_resized = zoom(array1, zoom_factors, order=1)
        return array1_resized, array2

def visualize_raster(data, title, output_filename=None, cmap='viridis', vmin=None, vmax=None):
    """
    Visualize a raster data array with better styling.
    
    Parameters:
        data (numpy.ndarray): 2D array to visualize
        title (str): Plot title
        output_filename (str): File name to save the plot
        cmap (str): Matplotlib colormap name
        vmin (float): Minimum value for color scaling
        vmax (float): Maximum value for color scaling
    """
    plt.figure(figsize=(10, 8))
    
    # Create a nicer styled plot
    ax = plt.gca()
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Set title with better font
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Remove ticks for cleaner look
    plt.tick_params(axis='both', which='both', bottom=False, top=False, 
                    left=False, right=False, labelbottom=False, labelleft=False)
    
    # Save plot if filename provided
    if output_filename:
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def advanced_visualization(result_dict, title, output_filename=None):
    """
    Create advanced visualization with multiple data sets.
    
    Parameters:
        result_dict (dict): Dictionary with named data arrays
                           'data' key is required, others are optional
        title (str): Plot title
        output_filename (str): File name to save the plot
    """
    # Expected keys: 'data' (required), 'comparison' (optional)
    if 'data' not in result_dict:
        raise ValueError("result_dict must contain 'data' key")
    
    data = result_dict['data']
    has_comparison = 'comparison' in result_dict
    
    # Create custom colormap for better visualization
    colors = [(0.1, 0.1, 0.6), (0.2, 0.5, 0.9), (0.98, 0.98, 0.82), (1, 0.5, 0), (0.8, 0.1, 0.1)]
    custom_cmap = LinearSegmentedColormap.from_list('urban_cmap', colors, N=256)
    
    if has_comparison:
        # Ensure data and comparison have same shape
        data, comparison = ensure_same_shape(data, result_dict['comparison'])
        
        # Create a figure with multiple subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot predicted data
        im0 = axs[0].imshow(data, cmap=custom_cmap)
        axs[0].set_title("Predicted", fontsize=12, fontweight='bold')
        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
        
        # Plot observed data (comparison)
        im1 = axs[1].imshow(comparison, cmap=custom_cmap)
        axs[1].set_title("Observed", fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
        
        # Plot difference
        diff = data - comparison
        im2 = axs[2].imshow(diff, cmap='RdBu_r', vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
        axs[2].set_title("Difference", fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
        
        # Remove ticks for cleaner look
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Main title
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
    else:
        # Simple plot with just the data
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        im = ax.imshow(data, cmap=custom_cmap)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Set title with better font
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Remove ticks for cleaner look
        plt.tick_params(axis='both', which='both', bottom=False, top=False, 
                        left=False, right=False, labelbottom=False, labelleft=False)
    
    # Save plot if filename provided
    if output_filename:
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def create_pedestrian_flow_animation(probability_results, time_periods, output_filename, fps=1):
    """
    Create an animation showing pedestrian flow over different time periods.
    
    Parameters:
        probability_results (dict): Dictionary mapping time periods to probability arrays
        time_periods (list): Ordered list of time period keys
        output_filename (str): File name to save the animation
        fps (int): Frames per second for the animation
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create custom colormap
    colors = [(0.1, 0.1, 0.6), (0.2, 0.5, 0.9), (0.98, 0.98, 0.82), (1, 0.5, 0), (0.8, 0.1, 0.1)]
    custom_cmap = LinearSegmentedColormap.from_list('urban_cmap', colors, N=256)
    
    # Find global min/max for consistent color scale
    vmin = min(np.min(probability_results[t]) for t in time_periods)
    vmax = max(np.max(probability_results[t]) for t in time_periods)
    
    # Initial plot
    im = ax.imshow(probability_results[time_periods[0]], cmap=custom_cmap, vmin=vmin, vmax=vmax)
    title = ax.text(0.5, 1.05, time_periods[0], transform=ax.transAxes, 
                   ha="center", va="bottom", fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Remove ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Animation update function
    def update(frame):
        time_period = time_periods[frame]
        im.set_array(probability_results[time_period])
        title.set_text(time_period)
        return [im, title]
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(time_periods), 
                                  interval=1000/fps, blit=True)
    
    # Save animation
    try:
        ani.save(output_filename, writer='pillow', fps=fps, dpi=150)
        print(f"Animation saved to {output_filename}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        # Try saving as separate images if GIF fails
        for i, time_period in enumerate(time_periods):
            plt.figure(figsize=(10, 8))
            plt.imshow(probability_results[time_period], cmap=custom_cmap, vmin=vmin, vmax=vmax)
            plt.title(time_period, fontsize=14, fontweight='bold')
            plt.axis('off')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(f"{output_filename.split('.')[0]}_{i}_{time_period}.png", dpi=150)
            plt.close()
        print("Saved individual frames as separate images.")