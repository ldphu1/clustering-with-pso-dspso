import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torchvision import datasets

from sklearn.metrics import silhouette_score, davies_bouldin_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.manifold import TSNE

from kmeans import KMeansWrapper

from ds_pso_clustering import *
from pso_clustering import *

import pandas as pd
import os


def export_log_to_excel(report_log, method_name, filename="clustering_results.xlsx"):
    """
    Lưu nhật ký chạy thử nghiệm ra file Excel.
    Mỗi phương pháp (method_name) sẽ được lưu thành một Sheet riêng trong cùng 1 file.
    """
    columns = ['K (Số cụm)', 'Silhouette', 'DBI', 'NMI', 'ARI']

    df = pd.DataFrame(report_log, columns=columns)

    df = df.round(4)

    if os.path.exists(filename):
        with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=method_name, index=False)
    else:
        with pd.ExcelWriter(filename, engine='openpyxl', mode='w') as writer:
            df.to_excel(writer, sheet_name=method_name, index=False)

    print(f"[v] Đã xuất báo cáo của {method_name} ra Excel (File: {filename} | Sheet: {method_name})")

def plot_results(features_2d, true_labels, labels_pred, fitness_history, method_name, k_val):
    """
    Hàm visualize
    """
    plt.figure(figsize=(18, 5))
    high_contrast_hex = [
        '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#800000',
        '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9'
    ]
    custom_cmap = ListedColormap(high_contrast_hex)

    # 1.Ground Truth
    plt.subplot(1, 3, 1)
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=true_labels, cmap=custom_cmap, edgecolor='k', s=20)
    plt.title("Thực tế (Ground Truth)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    # 2. Kết quả phân cụm
    plt.subplot(1, 3, 2)
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels_pred, cmap=custom_cmap, edgecolor='k', s=20)
    plt.title(f"Kết quả {method_name} (K = {k_val})")
    plt.xlabel("t-SNE 1")

    # 3. Đường cong hội tụ
    plt.subplot(1, 3, 3)
    plt.plot(fitness_history, color='red', linewidth=2)
    plt.title(f"Đường cong hội tụ ({method_name})")
    plt.xlabel("Vòng lặp (Epochs)")
    plt.ylabel("Fitness Score")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"bieu_do_{method_name}_K{k_val}.png", dpi=300)
    plt.show()


def evaluate_optimal_k(features_tensor, features_np, true_labels, method_class, method_name, k_min=2, k_max=15,
                       **kwargs):
    """
    Hàm Wrapper: Chạy vòng lặp tìm K tối ưu dựa trên DBI
    """
    print(f"\n{'=' * 50}")
    print(f"[*] BẮT ĐẦU TÌM K TỐI ƯU CHO: {method_name}")
    print(f"[*] Dải tìm kiếm: K từ {k_min} đến {k_max}")
    print(f"{'=' * 50}")

    best_k = -1
    best_combined_score = float('-inf')
    best_labels = None
    best_history = None

    report_log = []

    for k in range(k_min, k_max + 1):
        print(f"\n[>>>] Đang thử nghiệm với K = {k} [<<<]")

        # Khởi tạo mô hình với K hiện tại và các tham số kwargs được truyền vào
        model = method_class(n_clusters=k, **kwargs)
        labels, centroids, history = model.fit(features_tensor)

        # Đánh giá
        try:
            sil_score = silhouette_score(features_np, labels)
            dbi_score = davies_bouldin_score(features_np, labels)
            nmi_score = normalized_mutual_info_score(true_labels, labels)
            ari_score = adjusted_rand_score(true_labels, labels)

            print(
                f"      -> Silhouette: {sil_score:.4f} | DBI: {dbi_score:.4f} | NMI: {nmi_score:.4f} | ARI: {ari_score:.4f}")
            report_log.append((k, sil_score, dbi_score, nmi_score, ari_score))

            # Cập nhật Best K nếu DBI hiện tại tốt hơn

            combined_score = sil_score / (dbi_score + 1e-6)  # Tránh chia cho 0

            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_k = k
                best_labels = labels
                best_history = history

        except ValueError as e:
            print(f"[!] Bỏ qua K={k} do lỗi: {e}")

    print(f"\n[+] TỔNG KẾT {method_name}: K tối ưu tìm được là {best_k} (DBI: {best_combined_score:.4f})")
    return best_k, best_labels, best_history, report_log

def main():
    print("=== 1. TẢI VÀ CHUẨN BỊ DỮ LIỆU ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    stl10_test = datasets.STL10(root='./data', split='test', download=False)
    true_labels = np.array(stl10_test.labels)

    # Tải đặc trưng
    features_tensor = torch.load('image_features.pt')
    features_tensor = features_tensor.to(device).float()

    # Giảm chiều bằng PCA
    print(f"[*] Kích thước đặc trưng gốc: {features_tensor.shape}")
    print("[*] Đang tiến hành giảm chiều dữ liệu bằng PyTorch PCA...")

    n_components = 128
    U, S, V = torch.pca_lowrank(features_tensor, q=n_components)
    features_reduced_tensor = torch.matmul(features_tensor, V[:, :n_components])

    print(f"[+] Kích thước sau khi giảm chiều bằng PCA: {features_reduced_tensor.shape}")

    features_reduced_np = features_reduced_tensor.detach().cpu().numpy()

    # t-SNE dùng để nén dữ liệu xuống 2D phục vụ việc visualize
    print("[*] Đang chạy t-SNE để chuẩn bị tọa độ biểu đồ 2D...")
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    features_np_original = features_tensor.detach().cpu().numpy()
    features_2d = tsne.fit_transform(features_np_original)

    # Khai báo dải K cần test
    K_MIN = 5
    K_MAX = 15

    #K-means
    best_k, labels, history, log = evaluate_optimal_k(
        features_tensor=features_tensor,
        features_np=features_np_original,
        true_labels=true_labels,
        method_class=KMeansWrapper,
        method_name="K-Means Baseline",
        k_min=K_MIN,
        k_max=K_MAX
    )
    output(features_2d, true_labels, labels, history, "K-Means", best_k, log)

    #PSO
    best_k, labels, history, log = evaluate_optimal_k(
        features_tensor=features_reduced_tensor,
        features_np=features_reduced_np,
        true_labels=true_labels,
        method_class=PSO_GPU,
        method_name="PSO",
        k_min=K_MIN,
        k_max=K_MAX,
        # Các tham số kwargs
        n_particles=20,
        max_iter=500
    )
    output(features_2d, true_labels, labels, history, "PSO", best_k, log)

    # DS-PSO
    best_k, labels, history, log = evaluate_optimal_k(
        features_tensor=features_reduced_tensor,
        features_np=features_reduced_np,
        true_labels=true_labels,
        method_class=DSPSO_GPU,
        method_name="DS-PSO HYBRID",
        k_min=K_MIN,
        k_max=K_MAX,
        # Các tham số kwargs
        n_swarms=5,
        particles_per_swarm=20,
        max_iter=500,
        hybrid=False
    )
    output(features_2d, true_labels, labels, history, "DS-PSO", best_k, log)

    # hybrid
    best_k, labels, history, log = evaluate_optimal_k(
        features_tensor=features_reduced_tensor,
        features_np=features_reduced_np,
        true_labels=true_labels,
        method_class=DSPSO_GPU,
        method_name="DS-PSO HYBRID",
        k_min=K_MIN,
        k_max=K_MAX,
        # Các tham số kwargs
        n_swarms=5,
        particles_per_swarm=20,
        max_iter=500,
        hybrid=True
    )
    output(features_2d, true_labels, labels, history, "HYBRID", best_k, log)

def output(features_2d, true_labels, labels, history, method_name, best_k, log):
    export_log_to_excel(log, method_name)

    print("\n=== 2. XUẤT BIỂU ĐỒ CHO MÔ HÌNH TỐT NHẤT ===")
    plot_results(features_2d, true_labels, labels, history, method_name=method_name, k_val=best_k)

    print("\n=== BẢNG BÁO CÁO THỰC NGHIỆM ===")
    print(f"{'K':<5} | {'Silhouette':<15} | {'DBI':<15} | {'NMI':<15} | {'ARI':<15}")
    print("-" * 40)
    for k_val, sil, dbi, nmi, ari in log:
        print(f"{k_val:<5} | {sil:<15.4f} | {dbi:<15.4f} | {nmi:<15.4f} | {ari:<15.4f}")

if __name__ == "__main__":
    main()