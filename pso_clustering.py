import numpy as np
import torch
from sklearn.metrics import pairwise_distances_argmin_min
import torch.nn.functional as F
from early_stopping import EarlyStopping


class PSO_CPU:
    def __init__(self, n_clusters, n_particles=20, max_iter=100, w=0.72, c1=1.49, c2=1.49):
        """
        Khởi tạo thuật toán PSO Tiêu chuẩn (Phương pháp 1 trong Chương 2)
        :param n_clusters: Số lượng cụm cần phân
        :param n_particles: Số lượng hạt trong bầy (Swarm size)
        :param max_iter: Số vòng lặp tối đa
        """
        self.n_clusters = n_clusters
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def _initialize_particles(self, features):
        """Khởi tạo ngẫu nhiên vị trí tâm cụm từ các điểm dữ liệu"""
        n_samples, n_features = features.shape
        # Vị trí hạt: Ma trận 3D (n_particles, n_clusters, n_features)
        positions = np.array([features[np.random.choice(n_samples, self.n_clusters, replace=False)]
                              for _ in range(self.n_particles)])
        velocities = np.zeros_like(positions)
        return positions, velocities

    def _calculate_fitness(self, features, positions):
        """Tính SSE (Sum of Squared Errors) thuần Numpy để đồng bộ với bản GPU"""
        n_particles = positions.shape[0]
        fitness = np.zeros(n_particles)
        for i in range(n_particles):
            # Hàm này tính khoảng cách và trả về nhãn + khoảng cách nhỏ nhất
            _, distances = pairwise_distances_argmin_min(features, positions[i])
            fitness[i] = np.sum(distances ** 2)
        return fitness

    def fit(self, features):
        """Chạy thuật toán tối ưu phân cụm"""
        print(f"[*] Đang chạy PSO Tiêu chuẩn với quần thể {self.n_particles} hạt...")

        if not isinstance(features, np.ndarray):
            features = features.detach().cpu().numpy()

        # 1. Khởi tạo bầy đàn
        positions, velocities = self._initialize_particles(features)
        pbest_pos = positions.copy()
        pbest_fit = self._calculate_fitness(features, positions)

        # Tìm Global Best (gbest) ban đầu
        best_idx = np.argmin(pbest_fit)
        gbest_pos = pbest_pos[best_idx].copy()
        gbest_fit = pbest_fit[best_idx]

        early_stopper = EarlyStopping(patience=15, min_delta=1e-4)

        history = []

        # 2. Vòng lặp tối ưu chính
        for iteration in range(self.max_iter):
            # Tính random r1, r2 cho mỗi hạt (Vector hóa)
            r1 = np.random.rand(self.n_particles, 1, 1)
            r2 = np.random.rand(self.n_particles, 1, 1)

            # Cập nhật Vận tốc & Vị trí cho toàn bộ bầy cùng lúc
            velocities = (self.w * velocities +
                          self.c1 * r1 * (pbest_pos - positions) +
                          self.c2 * r2 * (gbest_pos - positions))
            positions = positions + velocities

            # Tính Fitness mới
            current_fit = self._calculate_fitness(features, positions)

            # Cập nhật Personal Best (pbest)
            better_mask = current_fit < pbest_fit
            pbest_fit[better_mask] = current_fit[better_mask]
            pbest_pos[better_mask] = positions[better_mask].copy()

            # Cập nhật Global Best (gbest)
            current_best_idx = np.argmin(pbest_fit)
            if pbest_fit[current_best_idx] < gbest_fit:
                gbest_fit = pbest_fit[current_best_idx]
                gbest_pos = pbest_pos[current_best_idx].copy()

            history.append(gbest_fit)

            if iteration % 30 == 0:
                print(f"Epoch {iteration}/{self.max_iter} - Global Best Fitness: {gbest_fit:.4f}")

            if early_stopper(current_loss=gbest_fit, epoch=iteration):
                break

        labels, _ = pairwise_distances_argmin_min(features, gbest_pos)

        print(f"[+] Hoàn thành PSO Tiêu chuẩn! Final Fitness: {gbest_fit:.4f}")
        return labels, gbest_pos, history


class PSO_GPU:
    def __init__(self, n_clusters, n_particles=20, max_iter=100, w=0.72, c1=1.49, c2=1.49):
        """
        Khởi tạo thuật toán PSO Tiêu chuẩn (Phiên bản tăng tốc bằng GPU / PyTorch)
        :param n_clusters: Số lượng cụm cần phân
        :param n_particles: Số lượng hạt trong bầy
        :param max_iter: Số vòng lặp tối đa
        """
        self.n_clusters = n_clusters
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _initialize_particles(self, features):
        """Khởi tạo ngẫu nhiên vị trí tâm cụm bằng Tensor"""
        n_samples, n_features = features.shape
        # Chọn ngẫu nhiên các điểm dữ liệu làm tâm cụm
        positions = torch.stack([
            features[torch.randperm(n_samples)[:self.n_clusters]]
            for _ in range(self.n_particles)
        ])
        velocities = torch.zeros_like(positions, device=self.device)
        return positions, velocities

    def _calculate_fitness(self, features, positions):
        # Lấy số lượng hạt đang được truyền vào hàm
        n_current_particles = positions.shape[0]

        # Khởi tạo tensor fitness với kích thước chuẩn
        fitness = torch.zeros(n_current_particles, device=self.device)

        # Vòng lặp chạy linh hoạt theo số hạt 20 hạt lúc đầu, hoặc 10 hạt lúc khởi tạo lại
        for i in range(n_current_particles):
            distances = torch.cdist(features, positions[i])
            min_dist, _ = torch.min(distances, dim=1)
            fitness[i] = torch.sum(min_dist ** 2)

        return fitness

    def fit(self, features):
        """Chạy thuật toán tối ưu phân cụm"""
        print(f"[*] Đang chạy PSO Tiêu chuẩn trên thiết bị: {self.device}")

        # Kiểm tra và ép kiểu dữ liệu đầu vào
        if isinstance(features, np.ndarray):
            features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        else:
            features_tensor = features.to(self.device).float()

        features_tensor = F.normalize(features_tensor, p=2, dim=1)

        # 1. Khởi tạo bầy đàn
        positions, velocities = self._initialize_particles(features_tensor)
        pbest_pos = positions.clone()
        pbest_fit = self._calculate_fitness(features_tensor, positions)

        # Tìm gbest ban đầu
        best_idx = torch.argmin(pbest_fit)
        gbest_pos = pbest_pos[best_idx].clone()
        gbest_fit = pbest_fit[best_idx].item()

        early_stopper = EarlyStopping(patience=15, min_delta=1e-4)

        history = []

        # 2. Vòng lặp tối ưu chính
        for iteration in range(self.max_iter):
            # Tính random r1, r2 bằng tensor trên GPU
            r1 = torch.rand((self.n_particles, 1, 1), device=self.device)
            r2 = torch.rand((self.n_particles, 1, 1), device=self.device)

            # Cập nhật Vận tốc & Vị trí (Vector hóa toàn bộ)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (pbest_pos - positions) +
                          self.c2 * r2 * (gbest_pos - positions))
            positions = positions + velocities

            positions = F.normalize(positions, p=2, dim=-1)

            # Tính Fitness mới
            current_fit = self._calculate_fitness(features_tensor, positions)

            # Cập nhật pbest
            better_mask = current_fit < pbest_fit
            pbest_fit[better_mask] = current_fit[better_mask]
            pbest_pos[better_mask] = positions[better_mask].clone()

            # Cập nhật gbest
            current_best_idx = torch.argmin(pbest_fit)
            if pbest_fit[current_best_idx].item() < gbest_fit:
                gbest_fit = pbest_fit[current_best_idx].item()
                gbest_pos = pbest_pos[current_best_idx].clone()

            history.append(gbest_fit)

            if iteration % 30 == 0:
                print(f"Epoch {iteration}/{self.max_iter} - Global Best Fitness: {gbest_fit:.4f}")

            if early_stopper(current_loss=gbest_fit, epoch=iteration):
                break

        # 3. Kết thúc và gán nhãn
        print(f"[+] Hoàn thành PSO Tiêu chuẩn! Final Fitness (Độ lỗi): {gbest_fit:.4f}")

        distances = torch.cdist(features_tensor, gbest_pos)
        labels = torch.argmin(distances, dim=1)

        return labels.cpu().numpy(), gbest_pos.cpu().numpy(), history