import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import torch.nn.functional as F
from early_stopping import EarlyStopping


class DSPSO_CPU:
    def __init__(self, n_clusters, n_swarms=3, particles_per_swarm=10, max_iter=100, w=0.72, c1=1.49, c2=1.49):
        self.n_clusters = n_clusters
        self.n_swarms = n_swarms
        self.n_particles = particles_per_swarm
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def _initialize_swarm(self, features, n_particles):
        """Khởi tạo bầy đàn bằng Numpy"""
        n_samples = features.shape[0]
        positions = np.array([
            features[np.random.choice(n_samples, self.n_clusters, replace=False)]
            for _ in range(n_particles)
        ])
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
        print(f"[*] Đang chạy DS-PSO (CPU) với {self.n_swarms} bầy, mỗi bầy {self.n_particles} hạt...")

        # Đảm bảo dữ liệu đầu vào là Numpy array và được chuẩn hóa L2
        if not isinstance(features, np.ndarray):
            features = features.cpu().numpy()

        # Chuẩn hóa L2 cho features
        features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)

        # --- KHỞI TẠO ĐA BẦY ĐÀN ĐỘC LẬP ---
        swarms_pos, swarms_vel = [], []
        swarms_pbest_pos, swarms_pbest_fit = [], []

        # Lưu trữ sbest cho từng bầy
        swarms_sbest_pos = []
        swarms_sbest_fit = np.zeros(self.n_swarms)

        global_best_pos = None
        global_best_fit = float('inf')

        for s in range(self.n_swarms):
            pos, vel = self._initialize_swarm(features_norm, self.n_particles)
            fit_val = self._calculate_fitness(features_norm, pos)

            swarms_pos.append(pos)
            swarms_vel.append(vel)
            swarms_pbest_pos.append(pos.copy())
            swarms_pbest_fit.append(fit_val.copy())

            # Tìm sbest cho bầy hiện tại
            best_idx = np.argmin(fit_val)
            swarms_sbest_pos.append(pos[best_idx].copy())
            swarms_sbest_fit[s] = fit_val[best_idx]

            # Cập nhật gbest
            if swarms_sbest_fit[s] < global_best_fit:
                global_best_fit = swarms_sbest_fit[s]
                global_best_pos = swarms_sbest_pos[s].copy()

        stagnation_counter = np.zeros(self.n_swarms)

        early_stopper = EarlyStopping(patience=15, min_delta=1e-4)

        history = []

        # --- VÒNG LẶP TỐI ƯU ---
        for iteration in range(self.max_iter):
            for s in range(self.n_swarms):
                r1, r2 = np.random.rand(), np.random.rand()

                # 1. TÍNH CHẤT SOCIAL: Di chuyển theo sbest của bầy
                swarms_vel[s] = (self.w * swarms_vel[s] +
                                 self.c1 * r1 * (swarms_pbest_pos[s] - swarms_pos[s]) +
                                 self.c2 * r2 * (swarms_sbest_pos[s] - swarms_pos[s]))
                swarms_pos[s] = swarms_pos[s] + swarms_vel[s]

                # Chuẩn hóa L2 cho tâm cụm (tránh tràn số và đồng bộ với không gian cosine)
                norms = np.linalg.norm(swarms_pos[s], axis=2, keepdims=True)
                # Tránh chia cho 0
                swarms_pos[s] = np.divide(swarms_pos[s], norms, out=np.zeros_like(swarms_pos[s]), where=norms != 0)

                current_fit = self._calculate_fitness(features_norm, swarms_pos[s])
                swarm_improved = False

                for p in range(self.n_particles):
                    if current_fit[p] < swarms_pbest_fit[s][p]:
                        swarms_pbest_fit[s][p] = current_fit[p]
                        swarms_pbest_pos[s][p] = swarms_pos[s][p].copy()

                        # Cập nhật Swarm Best
                        if current_fit[p] < swarms_sbest_fit[s]:
                            swarms_sbest_fit[s] = current_fit[p]
                            swarms_sbest_pos[s] = swarms_pos[s][p].copy()
                            swarm_improved = True

                            # Cập nhật Global Best
                            if current_fit[p] < global_best_fit:
                                global_best_fit = current_fit[p]
                                global_best_pos = swarms_pos[s][p].copy()

                # 2. TÍNH CHẤT DYNAMIC
                if not swarm_improved:
                    stagnation_counter[s] += 1
                else:
                    stagnation_counter[s] = 0

                # Nếu kẹt quá 10 epoch -> Khởi tạo lại 50% hạt kém nhất
                if stagnation_counter[s] > 10:
                    sorted_indices = np.argsort(swarms_pbest_fit[s])
                    worst_half_indices = sorted_indices[self.n_particles // 2:]

                    new_pos, new_vel = self._initialize_swarm(features_norm, len(worst_half_indices))

                    swarms_pos[s][worst_half_indices] = new_pos
                    swarms_vel[s][worst_half_indices] = new_vel

                    new_fit = self._calculate_fitness(features_norm, new_pos)
                    swarms_pbest_fit[s][worst_half_indices] = new_fit
                    swarms_pbest_pos[s][worst_half_indices] = new_pos.copy()

                    stagnation_counter[s] = 0

            # 3. TÍNH CHẤT SOCIAL: Bầy kém nhất học từ Gbest
            if iteration > 0 and iteration % 20 == 0:
                worst_swarm_idx = np.argmax(swarms_sbest_fit)
                if swarms_sbest_fit[worst_swarm_idx] > global_best_fit:
                    swarms_sbest_pos[worst_swarm_idx] = global_best_pos.copy()
                    swarms_sbest_fit[worst_swarm_idx] = global_best_fit
                    swarms_pos[worst_swarm_idx][0] = global_best_pos.copy()

            history.append(global_best_fit)
            if iteration % 30 == 0:
                print(f"Epoch {iteration}/{self.max_iter} - Global Best Fitness: {global_best_fit:.4f}")

            if early_stopper(current_loss=global_best_fit, epoch=iteration):
                break

        labels, _ = pairwise_distances_argmin_min(features_norm, global_best_pos)
        return labels, global_best_pos, history


class DSPSO_GPU:
    def __init__(self, n_clusters, n_swarms=5, particles_per_swarm=20, max_iter=150, w=0.72, c1=1.49, c2=1.49,
                 hybrid=True):
        self.n_clusters = n_clusters
        self.n_swarms = n_swarms
        self.n_particles = particles_per_swarm
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
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
        # Lấy số lượng hạt THỰC TẾ đang được truyền vào hàm
        n_current_particles = positions.shape[0]

        # Khởi tạo tensor fitness với kích thước chuẩn
        fitness = torch.zeros(n_current_particles, device=self.device)

        # Vòng lặp chạy 20 hạt lúc đầu, hoặc 10 hạt lúc khởi tạo lại
        for i in range(n_current_particles):
            distances = torch.cdist(features, positions[i])
            min_dist, _ = torch.min(distances, dim=1)
            fitness[i] = torch.sum(min_dist ** 2)

        return fitness

    def fit(self, features):
        print(f"[*] Đang chạy DS-PSO trên thiết bị: {self.device}")

        # Chuẩn bị dữ liệu
        if isinstance(features, np.ndarray):
            features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        else:
            features_tensor = features.to(self.device).float()
        features_tensor = F.normalize(features_tensor, p=2, dim=1)

        kmeans_centroids = None
        if self.hybrid:
            features_np = features_tensor.cpu().numpy()
            features_np_norm = features_np / np.linalg.norm(features_np, axis=1, keepdims=True)
            kmeans_init = KMeans(n_clusters=self.n_clusters, init='k-means++', max_iter=10, n_init=1)
            kmeans_init.fit(features_np_norm)
            kmeans_centroids = kmeans_init.cluster_centers_

        # --- KHỞI TẠO ĐA BẦY ĐÀN ĐỘC LẬP ---
        swarms_pos, swarms_vel = [], []
        swarms_pbest_pos, swarms_pbest_fit = [], []

        # Lưu trữ sbest của từng bầy riêng biệt
        swarms_sbest_pos = []
        swarms_sbest_fit = torch.zeros(self.n_swarms, device=self.device)

        global_best_pos = None
        global_best_fit = float('inf')

        for s in range(self.n_swarms):
            pos, vel = self._initialize_swarm(features_tensor, kmeans_centroids)
            fit_val = self._calculate_fitness(features_tensor, pos)

            swarms_pos.append(pos)
            swarms_vel.append(vel)
            swarms_pbest_pos.append(pos.clone())
            swarms_pbest_fit.append(fit_val.clone())

            # Tìm sbest cho bầy này
            best_idx = torch.argmin(fit_val)
            swarms_sbest_pos.append(pos[best_idx].clone())
            swarms_sbest_fit[s] = fit_val[best_idx].item()

            # Cập nhật gbest toàn cục
            if swarms_sbest_fit[s] < global_best_fit:
                global_best_fit = swarms_sbest_fit[s].item()
                global_best_pos = swarms_sbest_pos[s].clone()

        stagnation_counter = torch.zeros(self.n_swarms, device=self.device)

        early_stopper = EarlyStopping(patience=15, min_delta=1e-4)

        history = []

        # --- VÒNG LẶP TỐI ƯU ---
        for iteration in range(self.max_iter):
            for s in range(self.n_swarms):
                r1, r2 = torch.rand(1, device=self.device), torch.rand(1, device=self.device)

                # 1. TÍNH CHẤT SOCIAL: Hạt di chuyển theo pbest và SBEST của bầy nó, không lao thẳng về GBEST
                swarms_vel[s] = (self.w * swarms_vel[s] +
                                 self.c1 * r1 * (swarms_pbest_pos[s] - swarms_pos[s]) +
                                 self.c2 * r2 * (swarms_sbest_pos[s] - swarms_pos[s]))
                swarms_pos[s] = swarms_pos[s] + swarms_vel[s]
                swarms_pos[s] = F.normalize(swarms_pos[s], p=2, dim=-1)

                current_fit = self._calculate_fitness(features_tensor, swarms_pos[s])
                swarm_improved = False

                for p in range(self.n_particles):
                    if current_fit[p] < swarms_pbest_fit[s][p]:
                        swarms_pbest_fit[s][p] = current_fit[p]
                        swarms_pbest_pos[s][p] = swarms_pos[s][p].clone()

                        # Cập nhật sbest
                        if current_fit[p] < swarms_sbest_fit[s]:
                            swarms_sbest_fit[s] = current_fit[p].item()
                            swarms_sbest_pos[s] = swarms_pos[s][p].clone()
                            swarm_improved = True

                            # Cập nhật gbest
                            if current_fit[p] < global_best_fit:
                                global_best_fit = current_fit[p].item()
                                global_best_pos = swarms_pos[s][p].clone()

                # 2. TÍNH CHẤT DYNAMIC
                if not swarm_improved:
                    stagnation_counter[s] += 1
                else:
                    stagnation_counter[s] = 0

                # Nếu kẹt quá 10 epoch, reset 50% số hạt kém nhất của bầy này
                if stagnation_counter[s] > 10:
                    # Sắp xếp hạt theo fitness từ tốt đến xấu
                    sorted_indices = torch.argsort(swarms_pbest_fit[s])
                    worst_half_indices = sorted_indices[self.n_particles // 2:]

                    # Tạo vị trí mới ngẫu nhiên chỉ cho nửa kém nhất
                    n_samples = features_tensor.shape[0]
                    new_positions = torch.stack([
                        features_tensor[torch.randperm(n_samples)[:self.n_clusters]]
                        for _ in range(len(worst_half_indices))
                    ])

                    # Cập nhật lại vào bầy
                    swarms_pos[s][worst_half_indices] = new_positions
                    swarms_vel[s][worst_half_indices] = torch.zeros_like(new_positions, device=self.device)

                    # Tính lại fitness cho các hạt mới
                    new_fit = self._calculate_fitness(features_tensor, new_positions)
                    swarms_pbest_fit[s][worst_half_indices] = new_fit
                    swarms_pbest_pos[s][worst_half_indices] = new_positions.clone()

                    stagnation_counter[s] = 0

            # 3. TÍNH CHẤT SOCIAL: Sau mỗi 20 epoch, bầy kém nhất học hỏi từ Global Best
            if iteration > 0 and iteration % 20 == 0:
                worst_swarm_idx = torch.argmax(swarms_sbest_fit)
                if swarms_sbest_fit[worst_swarm_idx] > global_best_fit:
                    # Kéo thủ lĩnh của bầy kém nhất về vị trí của gbest
                    swarms_sbest_pos[worst_swarm_idx] = global_best_pos.clone()
                    swarms_sbest_fit[worst_swarm_idx] = global_best_fit
                    swarms_pos[worst_swarm_idx][0] = global_best_pos.clone()

            history.append(global_best_fit)

            if iteration % 30 == 0:
                print(f"Epoch {iteration}/{self.max_iter} - Global Best: {global_best_fit:.4f}")

            if early_stopper(current_loss=global_best_fit, epoch=iteration):
                break

        # --- GIAI ĐOẠN 2: TINH CHỈNH CỤC BỘ BẰNG K-MEANS (Hybrid) ---
        if self.hybrid:
            gbest_np = global_best_pos.cpu().numpy()
            gbest_np_norm = gbest_np / np.linalg.norm(gbest_np, axis=1, keepdims=True)
            kmeans_fine = KMeans(n_clusters=self.n_clusters, init=gbest_np_norm, max_iter=15, n_init=1)
            kmeans_fine.fit(features_np_norm)
            global_best_pos = torch.tensor(kmeans_fine.cluster_centers_, dtype=torch.float32, device=self.device)

        distances = torch.cdist(features_tensor, global_best_pos)
        labels = torch.argmin(distances, dim=1)

        return labels.cpu().numpy(), global_best_pos.cpu().numpy(), history