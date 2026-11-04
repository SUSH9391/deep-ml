import numpy as np

def segment_image(image, k: int, max_iters: int = 100):
    """
    Segment an RGB image using K-Means clustering on pixel colors.
    """
    # Convert input to numpy array to make manipulation clean
    img_array = np.array(image, dtype=np.float64)
    H, W, C = img_array.shape
    
    # 1. Flatten the image to a list of pixels in row-major order (H * W, 3)
    pixels = img_array.reshape(-1, C)
    n_pixels = pixels.shape[0]
    
    # 2. Initialize the k centroids as the first k pixels of this flattened list
    centroids = pixels[:k].copy()
    
    for _ in range(max_iters):
        # Assign each pixel to the centroid with the smallest Euclidean distance
        # Using broadcasting: pixels shape (N, 1, C), centroids shape (1, K, C)
        distances = np.linalg.norm(pixels[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        
        # Argmin handles tie-breaking automatically by picking the lowest index
        labels = np.argmin(distances, axis=1)
        
        # Update each centroid to be the mean of all pixels assigned to it
        new_centroids = centroids.copy()
        for i in range(k):
            cluster_pixels = pixels[labels == i]
            if len(cluster_pixels) > 0:
                new_centroids[i] = np.mean(cluster_pixels, axis=0)
            # If a cluster ends up with no assigned pixels, keep its previous centroid unchanged
            
        # Stop early if no centroid changes (exact equality)
        if np.array_equal(centroids, new_centroids):
            break
            
        centroids = new_centroids
        
    # 4. Build the segmented image by replacing each pixel with the final centroid it belongs to
    segmented_pixels = centroids[labels]
    segmented_image = segmented_pixels.reshape(H, W, C)
    
    # Round values to 4 decimal places and convert back to nested Python list
    return np.round(segmented_image, 4).tolist()
    