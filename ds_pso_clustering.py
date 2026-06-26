import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import torch.nn.functional as F
from early_stopping import EarlyStopping

class DSPSO_CPU:
    def __init__(self, n_clusters, max_iter=150, w_max=0.9, w_min=0.4, c1=2.0, c2=2.0,
                 v_max=1.0, k_neighbors=5, hybrid=True, **kwargs):
        self.n_clusters = n_clusters

        # Tương thích với tham số truyền vào từ main.py
        n_swarms = kwargs.get('n_swarms', 5)
        particles_per_swarm = kwargs.get('particles_per_swarm', 20)
        self.n_particles = kwargs.get('n_particles', n_swarms * particles_per_swarm)

        self.max_iter = max_iter
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.k_neighbors = min(k_neighbors, self.n_particles - 1)
        self.hybrid = hybrid

    def _initialize_swarm(self, features, kmeans_centroids=None):
        n_samples = features.shape[0]
        positions = np.array([
            features[np.random.choice(n_samples, self.n_clusters, replace=False)]
            for _ in range(self.n_particles)
        ])
        if self.hybrid and kmeans_centroids is not None:
            positions[0] = kmeans_centroids.copy()

        velocities = np.zeros_like(positions)
        return positions, velocities

    def _calculate_fitness(self, features, positions):
        n_current_particles = positions.shape[0]
        fitness = np.zeros(n_current_particles)
        for i in range(n_current_particles):
            _, distances = pairwise_distances_argmin_min(features, positions[i])
            fitness[i] = np.sum(distances ** 2)
        return fitness

    def fit(self, features):
        print(f"[*] Đang chạy DS-PSO (CPU) với quần thể {self.n_particles} hạt...")

        if not isinstance(features, np.ndarray):
            features = features.cpu().numpy()

        features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)

        kmeans_centroids = None
        if self.hybrid:
            kmeans_init = KMeans(n_clusters=self.n_clusters, init='k-means++', max_iter=10, n_init=1)
            kmeans_init.fit(features_norm)
            kmeans_centroids = kmeans_init.cluster_centers_

        # 1. Khởi tạo Bầy và Kinh nghiệm
        positions, velocities = self._initialize_swarm(features_norm, kmeans_centroids)
        pbest_pos = positions.copy()
        pbest_fit = self._calculate_fitness(features_norm, positions)

        best_idx = np.argmin(pbest_fit)
        gbest_pos = pbest_pos[best_idx].copy()
        gbest_fit = pbest_fit[best_idx]

        # 2. Khởi tạo Ma trận Mạng Xã Hội Động (Dynamic Social Network)
        social_network = np.random.randint(0, 2, (self.n_particles, self.n_particles)).astype(float)
        np.fill_diagonal(social_network, 0)  # Không tự kết nối với chính mình

        early_stopper = EarlyStopping(patience=15, min_delta=1e-4)
        history = []

        # --- VÒNG LẶP TỐI ƯU ---
        for iteration in range(self.max_iter):
            # Tính Trọng số quán tính thích nghi (Adaptive Inertia Weight)
            w = self.w_max - ((self.w_max - self.w_min) / self.max_iter) * iteration

            # --- CẬP NHẬT MẠNG XÃ HỘI (Algorithm 3) ---
            sorted_indices = np.argsort(pbest_fit)
            top_k_indices = sorted_indices[:self.k_neighbors]

            # Tính threshold để ngắt kết nối các hạt trì trệ (Dùng mức trung bình)
            threshold = np.mean(pbest_fit)

            # Kết nối với Top-k
            for idx in top_k_indices:
                social_network[:, idx] = 1.0
                social_network[idx, :] = 1.0

            # Ngắt kết nối với các cá thể yếu kém
            bad_indices = np.where(pbest_fit > threshold)[0]
            for idx in bad_indices:
                social_network[:, idx] = 0.0
                social_network[idx, :] = 0.0

            np.fill_diagonal(social_network, 0)

            # --- TÌM LOCAL BEST CHO TỪNG HẠT ---
            local_best_pos = np.zeros_like(pbest_pos)
            for i in range(self.n_particles):
                neighbors = np.where(social_network[i] == 1.0)[0]
                if len(neighbors) > 0:
                    best_neighbor_idx = neighbors[np.argmin(pbest_fit[neighbors])]
                    local_best_pos[i] = pbest_pos[best_neighbor_idx].copy()
                else:
                    # Nếu bị cô lập, sử dụng kinh nghiệm bản thân
                    local_best_pos[i] = pbest_pos[i].copy()

            # --- DI CHUYỂN HẠT ---
            r1 = np.random.rand(self.n_particles, 1, 1)
            r2 = np.random.rand(self.n_particles, 1, 1)

            # Phương trình cập nhật DS-PSO (sử dụng local_best thay cho pbest)
            velocities = (w * velocities +
                          self.c1 * r1 * (local_best_pos - positions) +
                          self.c2 * r2 * (gbest_pos - positions))

            # Kiểm soát biên vận tốc (Velocity Clamping)
            v_norms = np.linalg.norm(velocities, axis=(1, 2), keepdims=True)
            clamp_mask = v_norms > self.v_max
            velocities = np.where(clamp_mask, velocities * (self.v_max / (v_norms + 1e-8)), velocities)

            positions = positions + velocities

            # Ràng buộc không gian (Đưa về mặt cầu chuẩn hóa L2)
            norms = np.linalg.norm(positions, axis=2, keepdims=True)
            positions = np.divide(positions, norms, out=np.zeros_like(positions), where=norms != 0)

            # --- ĐÁNH GIÁ VÀ CẬP NHẬT ---
            current_fit = self._calculate_fitness(features_norm, positions)

            better_mask = current_fit < pbest_fit
            pbest_fit[better_mask] = current_fit[better_mask]
            pbest_pos[better_mask] = positions[better_mask].copy()

            current_best_idx = np.argmin(pbest_fit)
            if pbest_fit[current_best_idx] < gbest_fit:
                gbest_fit = pbest_fit[current_best_idx]
                gbest_pos = pbest_pos[current_best_idx].copy()

            history.append(gbest_fit)

            if iteration % 30 == 0:
                print(f"Epoch {iteration}/{self.max_iter} - Global Best Fitness: {gbest_fit:.4f}")

            if early_stopper(current_loss=gbest_fit, epoch=iteration):
                break

        # Tinh chỉnh cuối (Hybrid)
        if self.hybrid:
            kmeans_fine = KMeans(n_clusters=self.n_clusters, init=gbest_pos, max_iter=15, n_init=1)
            kmeans_fine.fit(features_norm)
            gbest_pos = kmeans_fine.cluster_centers_

        labels, _ = pairwise_distances_argmin_min(features_norm, gbest_pos)
        return labels, gbest_pos, history


class DSPSO_GPU:
    def __init__(self, n_clusters, max_iter=150, w_max=0.9, w_min=0.4, c1=2.0, c2=2.0,
                 v_max=0.5, k_neighbors=5, hybrid=True, **kwargs):
        self.n_clusters = n_clusters

        # Tương thích với tham số truyền vào từ main.py
        n_swarms = kwargs.get('n_swarms', 5)
        particles_per_swarm = kwargs.get('particles_per_swarm', 20)
        self.n_particles = kwargs.get('n_particles', n_swarms * particles_per_swarm)

        self.max_iter = max_iter
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.k_neighbors = min(k_neighbors, self.n_particles - 1)
        self.hybrid = hybrid
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _initialize_swarm(self, features, kmeans_centroids=None):
        n_samples = features.shape[0]
        positions = torch.stack([
            features[torch.randperm(n_samples)[:self.n_clusters]]
            for _ in range(self.n_particles)
        ])
        if self.hybrid and kmeans_centroids is not None:
            positions[0] = torch.tensor(kmeans_centroids, dtype=torch.float32, device=self.device)

        velocities = torch.zeros_like(positions, device=self.device)
        return positions, velocities

    def _calculate_fitness(self, features, positions):
        n_current_particles = positions.shape[0]
        fitness = torch.zeros(n_current_particles, device=self.device)
        for i in range(n_current_particles):
            distances = torch.cdist(features, positions[i])
            min_dist, _ = torch.min(distances, dim=1)
            fitness[i] = torch.sum(min_dist ** 2)
        return fitness

    def fit(self, features):
        print(f"[*] Đang chạy DS-PSO (Dynamic Social Network) trên thiết bị: {self.device}")

        # 1. Chuẩn bị dữ liệu
        if isinstance(features, np.ndarray):
            features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        else:
            features_tensor = features.to(self.device).float()
        features_tensor = F.normalize(features_tensor, p=2, dim=1)

        kmeans_centroids = None
        if self.hybrid:
            features_np = features_tensor.cpu().numpy()
            kmeans_init = KMeans(n_clusters=self.n_clusters, init='k-means++', max_iter=10, n_init=1)
            kmeans_init.fit(features_np)
            kmeans_centroids = kmeans_init.cluster_centers_

        # 2. Khởi tạo Quần thể
        positions, velocities = self._initialize_swarm(features_tensor, kmeans_centroids)
        pbest_pos = positions.clone()
        pbest_fit = self._calculate_fitness(features_tensor, positions)

        best_idx = torch.argmin(pbest_fit)
        gbest_pos = pbest_pos[best_idx].clone()
        gbest_fit = pbest_fit[best_idx].item()

        # Khởi tạo Ma trận Mạng Xã Hội Động (Algorithm 2)
        social_network = torch.randint(0, 2, (self.n_particles, self.n_particles), device=self.device).float()
        social_network.fill_diagonal_(0)

        early_stopper = EarlyStopping(patience=15, min_delta=1e-4)
        history = []

        # --- VÒNG LẶP TỐI ƯU ---
        for iteration in range(self.max_iter):

            # Tính Trọng số quán tính thích nghi (Adaptive Inertia Weight)
            w = self.w_max - ((self.w_max - self.w_min) / self.max_iter) * iteration

            # --- CẬP NHẬT MẠNG XÃ HỘI (Algorithm 3) ---
            sorted_indices = torch.argsort(pbest_fit)
            top_k_indices = sorted_indices[:self.k_neighbors]

            # Ngưỡng ngắt kết nối (sử dụng mức fitness trung bình)
            threshold = torch.mean(pbest_fit)

            # Kết nối với Top-k
            for idx in top_k_indices:
                social_network[:, idx] = 1.0
                social_network[idx, :] = 1.0

            # Ngắt kết nối với các hạt trì trệ
            bad_indices = torch.where(pbest_fit > threshold)[0]
            for idx in bad_indices:
                social_network[:, idx] = 0.0
                social_network[idx, :] = 0.0

            social_network.fill_diagonal_(0)

            # --- TÌM LOCAL BEST CHO TỪNG HẠT ---
            local_best_pos = torch.zeros_like(pbest_pos)
            for i in range(self.n_particles):
                neighbors = torch.where(social_network[i] == 1.0)[0]
                if len(neighbors) > 0:
                    best_neighbor_idx = neighbors[torch.argmin(pbest_fit[neighbors])]
                    local_best_pos[i] = pbest_pos[best_neighbor_idx].clone()
                else:
                    # Nếu bị cô lập, sử dụng kinh nghiệm bản thân
                    local_best_pos[i] = pbest_pos[i].clone()

            # --- DI CHUYỂN HẠT ---
            r1 = torch.rand((self.n_particles, 1, 1), device=self.device)
            r2 = torch.rand((self.n_particles, 1, 1), device=self.device)

            # Phương trình cập nhật vận tốc (Algorithm 4)
            velocities = (w * velocities +
                          self.c1 * r1 * (local_best_pos - positions) +
                          self.c2 * r2 * (gbest_pos - positions))

            # Velocity Clamping (Kiểm soát biên vận tốc)
            v_norms = torch.norm(velocities, p=2, dim=(1, 2), keepdim=True)
            clamp_mask = (v_norms > self.v_max).expand_as(velocities)
            velocities = torch.where(clamp_mask, velocities * (self.v_max / (v_norms + 1e-8)), velocities)

            # Cập nhật vị trí
            positions = positions + velocities
            positions = F.normalize(positions, p=2, dim=-1)

            # --- ĐÁNH GIÁ VÀ CẬP NHẬT KINH NGHIỆM ---
            current_fit = self._calculate_fitness(features_tensor, positions)

            better_mask = current_fit < pbest_fit
            pbest_fit[better_mask] = current_fit[better_mask]
            pbest_pos[better_mask] = positions[better_mask].clone()

            current_best_idx = torch.argmin(pbest_fit)
            if pbest_fit[current_best_idx].item() < gbest_fit:
                gbest_fit = pbest_fit[current_best_idx].item()
                gbest_pos = pbest_pos[current_best_idx].clone()

            history.append(gbest_fit)

            if iteration % 30 == 0:
                print(f"Epoch {iteration}/{self.max_iter} - Global Best Fitness: {gbest_fit:.4f}")

            if early_stopper(current_loss=gbest_fit, epoch=iteration):
                break

        # (Hybrid) Tinh chỉnh cục bộ bằng K-Means ở cuối quá trình
        if self.hybrid:
            gbest_np = gbest_pos.cpu().numpy()
            kmeans_fine = KMeans(n_clusters=self.n_clusters, init=gbest_np, max_iter=15, n_init=1)
            kmeans_fine.fit(features_tensor.cpu().numpy())
            gbest_pos = torch.tensor(kmeans_fine.cluster_centers_, dtype=torch.float32, device=self.device)

        distances = torch.cdist(features_tensor, gbest_pos)
        labels = torch.argmin(distances, dim=1)

        return labels.cpu().numpy(), gbest_pos.cpu().numpy(), history