import numpy as np
import matplotlib.pyplot as plt
# import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from scipy.spatial.transform import Rotation
from scipy.stats import gaussian_kde
import pandas as pd

class SpatialRotationVisualizer:
    def __init__(self, data):
        """
        Initialize with 6D data where:
        - data[:, 0:3] are (x, y, z) spatial coordinates
        - data[:, 3:6] are axis-angle rotation vectors
        """
        self.data = data
        self.n_points = data.shape[0]
        self.spatial = data[:, 0:3]
        self.rotations = data[:, 3:6]
        
        # Convert axis-angle to other rotation representations
        self._process_rotations()
        
    def _process_rotations(self):
        """Convert axis-angle to quaternions and rotation matrices"""
        # Handle zero rotations (avoid division by zero)
        angles = np.linalg.norm(self.rotations, axis=1)
        axes = np.zeros_like(self.rotations)
        
        # Only process non-zero rotations
        nonzero_mask = angles > 1e-6
        axes[nonzero_mask] = self.rotations[nonzero_mask] / angles[nonzero_mask, np.newaxis]
        
        # Convert to scipy Rotation objects
        rot_objects = Rotation.from_rotvec(self.rotations)
        self.quaternions = rot_objects.as_quat()  # (x, y, z, w)
        self.rotation_matrices = rot_objects.as_matrix()
        
        # Store axis and angle separately for visualization
        self.rotation_axes = axes
        self.rotation_angles = angles
    
    def sample_data(self, n_samples=5000):
        """Sample subset of data for faster visualization"""
        if self.n_points <= n_samples:
            return np.arange(self.n_points)
        
        # Stratified sampling to maintain distribution
        indices = np.random.choice(self.n_points, n_samples, replace=False)
        return indices
    
    def visualize_spatial_3d(self, sample_size=5000, method='scatter'):
        """Visualize spatial distribution in 3D"""
        indices = self.sample_data(sample_size)
        spatial_sample = self.spatial[indices]
        
        if method == 'scatter':
            # Basic 3D scatter plot
            fig = go.Figure(data=go.Scatter3d(
                x=spatial_sample[:, 0],
                y=spatial_sample[:, 1],
                z=spatial_sample[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=self.rotation_angles[indices],
                    colorscale='Viridis',
                    colorbar=dict(title="Rotation Angle (rad)"),
                    opacity=0.6
                )
            ))
            fig.update_layout(
                title='3D Spatial Distribution (colored by rotation angle)',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'
                )
            )
            
        elif method == 'density':
            # Density-based visualization using hexbin-like approach
            # Create a 3D histogram
            hist, edges = np.histogramdd(spatial_sample, bins=20)
            
            # Find non-zero bins
            indices_3d = np.where(hist > 0)
            x_centers = (edges[0][indices_3d[0]] + edges[0][indices_3d[0] + 1]) / 2
            y_centers = (edges[1][indices_3d[1]] + edges[1][indices_3d[1] + 1]) / 2
            z_centers = (edges[2][indices_3d[2]] + edges[2][indices_3d[2] + 1]) / 2
            values = hist[indices_3d]
            
            fig = go.Figure(data=go.Scatter3d(
                x=x_centers,
                y=y_centers,
                z=z_centers,
                mode='markers',
                marker=dict(
                    size=np.sqrt(values) * 2,
                    color=values,
                    colorscale='Hot',
                    colorbar=dict(title="Point Density"),
                    opacity=0.8
                )
            ))
            fig.update_layout(title='3D Spatial Density Distribution')
        
        return fig
    
    def visualize_rotations_sphere(self, sample_size=2000):
        """Visualize rotation axes on unit sphere"""
        indices = self.sample_data(sample_size)
        axes_sample = self.rotation_axes[indices]
        angles_sample = self.rotation_angles[indices]
        
        # Convert to spherical coordinates
        phi = np.arctan2(axes_sample[:, 1], axes_sample[:, 0])
        theta = np.arccos(np.clip(axes_sample[:, 2], -1, 1))
        
        fig = go.Figure()
        
        # Add unit sphere surface
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 25)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            opacity=0.1,
            colorscale='Blues',
            showscale=False
        ))
        
        # Add rotation axes
        fig.add_trace(go.Scatter3d(
            x=axes_sample[:, 0],
            y=axes_sample[:, 1],
            z=axes_sample[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=angles_sample,
                colorscale='Plasma',
                colorbar=dict(title="Rotation Angle (rad)"),
            ),
            name='Rotation Axes'
        ))
        
        fig.update_layout(
            title='Rotation Axes on Unit Sphere (colored by angle)',
            scene=dict(
                xaxis_title='X axis',
                yaxis_title='Y axis',
                zaxis_title='Z axis',
                aspectmode='cube'
            )
        )
        
        return fig
    
    def visualize_combined_pairwise(self, sample_size=5000):
        """Create pairwise scatter plots of all 6 dimensions"""
        indices = self.sample_data(sample_size)
        
        # Combine spatial and rotation data
        combined_data = np.column_stack([
            self.spatial[indices],
            self.rotation_axes[indices],
            self.rotation_angles[indices]
        ])
        
        labels = ['X', 'Y', 'Z', 'Rot_Axis_X', 'Rot_Axis_Y', 'Rot_Axis_Z', 'Rot_Angle']
        
        # Create DataFrame for seaborn
        df = pd.DataFrame(combined_data, columns=labels)
        
        # Create pairplot
        g = sns.PairGrid(df, diag_sharey=False, height=2.5)
        g.map_upper(sns.scatterplot, alpha=0.5, s=1)
        g.map_lower(sns.scatterplot, alpha=0.5, s=1)
        g.map_diag(sns.histplot)
        
        g.figure.suptitle('Pairwise Relationships in 6D Data', y=1.02)
        g.figure.tight_layout()
        
        return g
    
    def dimensionality_reduction(self, method='umap', sample_size=10000):
        """Apply dimensionality reduction to 6D data"""
        indices = self.sample_data(sample_size)
        
        # Combine spatial and quaternion data (quaternions are better for ML)
        combined_data = np.column_stack([
            self.spatial[indices],
            self.quaternions[indices]
        ])
        
        if method == 'pca':
            reducer = PCA(n_components=2)
            embedding = reducer.fit_transform(combined_data)
            title = f'PCA 2D Embedding (explained variance: {reducer.explained_variance_ratio_.sum():.2f})'
            
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            embedding = reducer.fit_transform(combined_data)
            title = 't-SNE 2D Embedding'
            
        elif method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedding = reducer.fit_transform(combined_data)
            title = 'UMAP 2D Embedding'
        
        # Color by rotation angle
        fig = plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embedding[:, 0], 
            embedding[:, 1], 
            c=self.rotation_angles[indices],
            cmap='viridis',
            alpha=0.6,
            s=1
        )
        plt.colorbar(scatter, label='Rotation Angle (rad)')
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
        return fig, embedding
    
    def parallel_coordinates_plot(self, sample_size=2000):
        """Create parallel coordinates plot"""
        indices = self.sample_data(sample_size)
        
        # Normalize all dimensions for parallel coordinates
        data_normalized = np.column_stack([
            self.spatial[indices],
            self.rotation_axes[indices],
            self.rotation_angles[indices]
        ])
        
        # Normalize each dimension to [0, 1]
        data_normalized = (data_normalized - data_normalized.min(axis=0)) / \
                         (data_normalized.max(axis=0) - data_normalized.min(axis=0))
        
        dimensions = [
            dict(label='X', values=data_normalized[:, 0]),
            dict(label='Y', values=data_normalized[:, 1]),
            dict(label='Z', values=data_normalized[:, 2]),
            dict(label='Rot_Axis_X', values=data_normalized[:, 3]),
            dict(label='Rot_Axis_Y', values=data_normalized[:, 4]),
            dict(label='Rot_Axis_Z', values=data_normalized[:, 5]),
            dict(label='Rot_Angle', values=data_normalized[:, 6])
        ]
        
        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=self.rotation_angles[indices],
                colorscale='Viridis',
                colorbar=dict(title="Rotation Angle")
            ),
            dimensions=dimensions
        ))
        
        fig.update_layout(title='Parallel Coordinates Plot (6D + angle)')
        return fig

# Example usage and demo data generation
def generate_demo_data(n_points=50000):
    """Generate synthetic 6D data for demonstration"""
    np.random.seed(42)
    
    # Create clustered spatial data
    cluster_centers = np.array([[0, 0, 0], [5, 5, 5], [-3, 2, 4]])
    spatial_data = []
    
    for center in cluster_centers:
        n_cluster = n_points // len(cluster_centers)
        cluster_data = np.random.multivariate_normal(
            center, np.eye(3) * 0.5, n_cluster
        )
        spatial_data.append(cluster_data)
    
    spatial_data = np.vstack(spatial_data)
    
    # Create rotation data with some structure
    # Rotations somewhat correlated with spatial position
    rotation_data = np.random.randn(n_points, 3) * 0.5
    
    # Add some correlation between spatial and rotational components
    rotation_data += spatial_data * 0.1
    
    return np.column_stack([spatial_data, rotation_data])

# Demo usage
if __name__ == "__main__":
    # Generate or load your data
    # data = np.load('your_6d_data.npy')  # Replace with your data loading
    # data = generate_demo_data(9999)  # Demo data

    # read from a npz file
    data = np.load('/scratch/yz12129/hrdt_pretrain/egodex_first_frames.npz')['first_frames']  # Replace

    data = data[:, 7:] # use last 7 dimensions for left hand
    
    # Create visualizer
    viz = SpatialRotationVisualizer(data)
    
    # Generate various visualizations
    fig1 = viz.visualize_spatial_3d(method='scatter', sample_size=50000)
    fig1.show()
    # save go fig
    # fig1.write_image("spatial_3d_scatter.png")
    
    fig2 = viz.visualize_rotations_sphere()
    fig2.show()
    # fig2.write_image("rotations_sphere.png")
    
    fig3 = viz.visualize_combined_pairwise()
    fig3.figure.show()
    fig3.figure.savefig("combined_pairwise.png")
    
    fig4, embedding = viz.dimensionality_reduction(method='umap')
    plt.show()
    fig4.savefig("dimensionality_reduction_umap.png")
    
    fig5 = viz.parallel_coordinates_plot()
    fig5.show()
    # fig5.write_image("parallel_coordinates.png")