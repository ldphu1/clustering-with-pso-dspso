import torch
from sklearn.cluster import KMeans

class KMeansWrapper:
    def __init__(self, n_clusters, random_state=42, **kwargs):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, init='k-means++', n_init='auto', **kwargs)

    def fit(self, features_tensor):
        if torch.is_tensor(features_tensor):
            features_np = features_tensor.detach().cpu().numpy()
        else:
            features_np = features_tensor

        self.model.fit(features_np)

        labels = self.model.labels_
        centroids = self.model.cluster_centers_

        dummy_history = [self.model.inertia_]

        return labels, centroids, dummy_history