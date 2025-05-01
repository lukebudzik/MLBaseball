from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import BaseballDataVisualization1D as bbData

# Select only numeric columns from cleaned dataset
numeric_cols = ['release_speed','release_pos_x','release_pos_z','pfx_x', 'pfx_z', 'plate_x',
    'plate_z','vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'sz_top','sz_bot', 'effective_speed',
    'release_spin_rate', 'release_extension','release_pos_y','bat_speed','swing_length',
    'api_break_z_with_gravity','api_break_x_arm', 'api_break_x_batter_in', 'arm_angle']
df = bbData.csv_to_df()
X_numeric = df[numeric_cols].dropna()  # Drop rows with NaNs for simplicity
X_scaled = StandardScaler().fit_transform(X_numeric)

pca = PCA(n_components=2)  # Reduce to 2 dimensions
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Pitching Data')
plt.grid(True)
plt.show()


pca_full = PCA().fit(X_scaled)
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(pca_full.explained_variance_ratio_)+1), pca_full.explained_variance_ratio_, marker='o')
plt.title('Explained Variance by Principal Component')
plt.xlabel('Component #')
plt.ylabel('Variance Ratio')
plt.grid(True)
plt.show()