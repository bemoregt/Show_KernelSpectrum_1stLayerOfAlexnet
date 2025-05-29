import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QLabel, QWidget, QComboBox
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt
from scipy.ndimage import zoom
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# 메인 모듈 체크
if __name__ == "__main__":
    # 장치 설정 (MPS 사용)
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"사용 장치: {device}")
    
    # PyQt5 GUI 클래스 정의
    class WeightVisualizerWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("딥러닝 모델 첫 번째 컨볼루션 레이어 가중치 시각화")
            self.setGeometry(100, 100, 1200, 800)
            
            # 중앙 위젯 설정
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            
            # 그리드 레이아웃 설정
            grid_layout = QGridLayout(central_widget)
            
            # 시각화 모드 선택 콤보 박스
            self.mode_combo = QComboBox()
            self.mode_combo.addItems(["실제 가중치", "고대비 가중치", "컬러 채널 시각화", "푸리에 진폭 스펙트럼"])
            self.mode_combo.setCurrentIndex(2)  # 기본값으로 컬러 채널 시각화 모드 설정
            self.mode_combo.currentIndexChanged.connect(self.update_visualization)
            grid_layout.addWidget(self.mode_combo, 0, 0, 1, 8)
            
            # 모델 선택 콤보 박스
            self.model_combo = QComboBox()
            self.model_combo.addItems([
                "AlexNet", "VGG16", "ResNet50", 
                "LeNet5", "DenseNet121", "MobileNetV2", 
                "GoogLeNet", "InceptionV3"
            ])
            self.model_combo.currentIndexChanged.connect(self.change_model)
            grid_layout.addWidget(self.model_combo, 1, 0, 1, 8)
            
            # 필터 라벨 (첫 번째 레이어는 모델마다 필터 수가 다름)
            self.filter_labels = []
            # 기본적으로 64개의 필터를 8x8 그리드로 표시
            for i in range(64):
                row, col = divmod(i, 8)
                label = QLabel()
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setMinimumSize(120, 120)
                grid_layout.addWidget(label, row + 2, col)
                self.filter_labels.append(label)
            
            # 설명 라벨
            self.info_label = QLabel("ImageNet으로 사전 학습된 모델의 첫 번째 컨볼루션 레이어 가중치")
            self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid_layout.addWidget(self.info_label, 10, 0, 1, 8)
            
            # 모델과 가중치 초기화
            self.model = None
            self.current_weights = None
            
            # 클러스터 색상 정의 (최대 10개 클러스터)
            self.cluster_colors = [
                (255, 0, 0),    # 빨강
                (0, 255, 0),    # 초록
                (0, 0, 255),    # 파랑
                (255, 255, 0),  # 노랑
                (255, 0, 255),  # 마젠타
                (0, 255, 255),  # 시안
                (255, 128, 0),  # 주황
                (128, 0, 255),  # 보라
                (255, 192, 203),# 핑크
                (128, 128, 128) # 회색
            ]
            
            self.change_model(0)  # 기본 모델(AlexNet) 로드
            
        def change_model(self, index):
            # 선택된 모델 로드
            model_name = self.model_combo.currentText()
            
            try:
                if model_name == "AlexNet":
                    self.model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
                    # AlexNet의 첫 번째 컨볼루션 레이어 가중치 가져오기
                    self.current_weights = self.model.features[0].weight
                    
                elif model_name == "VGG16":
                    self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
                    # VGG16의 첫 번째 컨볼루션 레이어 가중치 가져오기
                    self.current_weights = self.model.features[0].weight
                
                elif model_name == "ResNet50":
                    self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                    # ResNet50의 첫 번째 컨볼루션 레이어 가중치 가져오기
                    self.current_weights = self.model.conv1.weight
                
                elif model_name == "LeNet5":
                    # LeNet5 모델 정의 (토치비전에는 없으므로 직접 구현)
                    class LeNet5(nn.Module):
                        def __init__(self):
                            super(LeNet5, self).__init__()
                            self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
                            self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
                            self.fc1 = nn.Linear(16*4*4, 120)
                            self.fc2 = nn.Linear(120, 84)
                            self.fc3 = nn.Linear(84, 10)
                            
                        def forward(self, x):
                            x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv1(x)), 2)
                            x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv2(x)), 2)
                            x = x.view(-1, 16*4*4)
                            x = torch.nn.functional.relu(self.fc1(x))
                            x = torch.nn.functional.relu(self.fc2(x))
                            x = self.fc3(x)
                            return x
                    
                    # LeNet5 모델 초기화 (ImageNet 학습 가중치는 없음)
                    self.model = LeNet5()
                    # 첫 번째 컨볼루션 레이어 가중치 가져오기
                    self.current_weights = self.model.conv1.weight
                    # 입력이 흑백이므로 확장하여 RGB 시각화 가능하게 함
                    if self.current_weights.shape[1] == 1:
                        weights_expanded = torch.zeros((self.current_weights.shape[0], 3, 
                                                      self.current_weights.shape[2], 
                                                      self.current_weights.shape[3]))
                        for i in range(3):  # RGB 채널에 동일한 흑백 가중치 복제
                            weights_expanded[:, i] = self.current_weights[:, 0]
                        self.current_weights = weights_expanded
                
                elif model_name == "DenseNet121":
                    self.model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
                    # DenseNet121의 첫 번째 컨볼루션 레이어 가중치 가져오기
                    self.current_weights = self.model.features.conv0.weight
                
                elif model_name == "MobileNetV2":
                    self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
                    # MobileNetV2의 첫 번째 컨볼루션 레이어 가중치 가져오기
                    self.current_weights = self.model.features[0][0].weight
                
                elif model_name == "GoogLeNet":
                    self.model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
                    # GoogLeNet의 첫 번째 컨볼루션 레이어 가중치 가져오기
                    self.current_weights = self.model.conv1.conv.weight
                
                elif model_name == "InceptionV3":
                    self.model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
                    # InceptionV3의 첫 번째 컨볼루션 레이어 가중치 가져오기
                    self.current_weights = self.model.Conv2d_1a_3x3.conv.weight
                
                # 필터 정보 업데이트
                self.info_label.setText(f"{model_name} - 첫 번째 컨볼루션 레이어: {self.current_weights.shape[0]} 필터, "
                                      f"입력 채널: {self.current_weights.shape[1]}, "
                                      f"커널 크기: {self.current_weights.shape[2]}x{self.current_weights.shape[3]}")
                
                # UI 업데이트 (필터 수가 변경될 수 있음)
                self.update_visualization()
                
            except Exception as e:
                # 오류 발생 시 정보 표시
                self.info_label.setText(f"오류 발생: {str(e)}")
                print(f"모델 로드 오류: {str(e)}")
        
        def resize_kernel_smooth(self, input_img, target_size=(16, 16)):
            """스무딩 방식으로 커널을 리사이즈"""
            current_h, current_w = input_img.shape
            target_h, target_w = target_size
            
            # zoom factor 계산
            zoom_h = target_h / current_h
            zoom_w = target_w / current_w
            
            # scipy.ndimage.zoom을 사용하여 스무딩 리사이즈 (3차 스플라인 보간)
            resized_img = zoom(input_img, (zoom_h, zoom_w), order=3)
            
            return resized_img
        
        def compute_fft_magnitude(self, input_img):
            """커널을 16x16으로 리사이즈한 후 2D 푸리에 변환을 계산하고 진폭 스펙트럼을 반환"""
            # 커널을 16x16으로 스무딩 리사이즈
            resized_img = self.resize_kernel_smooth(input_img, target_size=(16, 16))
            
            # NumPy의 FFT 함수 사용
            fft_result = np.fft.fft2(resized_img)
            # 주파수 0(DC)이 중앙에 오도록 시프트
            fft_shifted = np.fft.fftshift(fft_result)
            # 진폭 계산 (복소수의 절대값)
            magnitude = np.abs(fft_shifted)
            # 로그 스케일로 변환 (동적 범위 압축)
            magnitude = np.log1p(magnitude)  # log(1+x)
            return magnitude
        
        def resize_spectrum_to_32x32(self, spectrum):
            """스펙트럼을 32x32로 보간 리사이징"""
            current_h, current_w = spectrum.shape
            zoom_h = 32 / current_h
            zoom_w = 32 / current_w
            
            # 3차 스플라인 보간으로 32x32로 리사이즈
            resized_spectrum = zoom(spectrum, (zoom_h, zoom_w), order=3)
            return resized_spectrum
        
        def compute_cosine_similarity_clustering(self, spectrums, threshold=0.7):
            """스펙트럼들 간의 코사인 유사도를 계산하고 클러스터링"""
            # 스펙트럼들을 1차원으로 평탄화
            flattened_spectrums = [spectrum.flatten() for spectrum in spectrums]
            
            # 코사인 유사도 계산
            similarity_matrix = cosine_similarity(flattened_spectrums)
            
            # 코사인 거리 = 1 - 코사인 유사도 (0과 2 사이 값)
            distance_matrix = 1 - similarity_matrix
            
            # 거리 행렬이 음수가 되지 않도록 클리핑 (안전장치)
            distance_matrix = np.clip(distance_matrix, 0, 2)
            
            # 대각선을 0으로 설정 (자기 자신과의 거리는 0)
            np.fill_diagonal(distance_matrix, 0)
            
            # DBSCAN 클러스터링 (eps는 1-threshold로 설정)
            eps = 1 - threshold
            
            try:
                dbscan = DBSCAN(eps=eps, min_samples=2, metric='precomputed')
                cluster_labels = dbscan.fit_predict(distance_matrix)
            except Exception as e:
                print(f"DBSCAN 오류: {e}")
                # 오류 발생 시 간단한 임계값 기반 클러스터링으로 대체
                cluster_labels = self.simple_threshold_clustering(similarity_matrix, threshold)
            
            return cluster_labels
        
        def simple_threshold_clustering(self, similarity_matrix, threshold):
            """간단한 임계값 기반 클러스터링"""
            n_samples = similarity_matrix.shape[0]
            cluster_labels = np.full(n_samples, -1)
            current_cluster = 0
            
            for i in range(n_samples):
                if cluster_labels[i] == -1:  # 아직 클러스터에 배정되지 않은 경우
                    # 새 클러스터 시작
                    cluster_labels[i] = current_cluster
                    
                    # 유사한 다른 샘플들을 같은 클러스터에 배정
                    for j in range(i + 1, n_samples):
                        if cluster_labels[j] == -1 and similarity_matrix[i, j] >= threshold:
                            cluster_labels[j] = current_cluster
                    
                    current_cluster += 1
            
            return cluster_labels
        
        def draw_border_on_pixmap(self, pixmap, color, border_width=3):
            """QPixmap에 컬러 테두리를 그리기"""
            # 새로운 QPixmap 생성 (원본 수정 방지)
            bordered_pixmap = QPixmap(pixmap.size())
            bordered_pixmap.fill(Qt.GlobalColor.transparent)
            
            # QPainter를 사용하여 테두리 그리기
            painter = QPainter(bordered_pixmap)
            
            # 원본 이미지를 먼저 그리기
            painter.drawPixmap(0, 0, pixmap)
            
            # 테두리 색상 설정
            from PyQt5.QtGui import QColor
            pen = QPen()
            qcolor = QColor(color[0], color[1], color[2])
            pen.setColor(qcolor)
            pen.setWidth(border_width)
            painter.setPen(pen)
            
            # 테두리 그리기
            rect = bordered_pixmap.rect()
            painter.drawRect(rect.adjusted(border_width//2, border_width//2, 
                                         -(border_width//2), -(border_width//2)))
            painter.end()
            
            return bordered_pixmap
        
        def update_visualization(self):
            if self.current_weights is None:
                return
            
            # 가중치를 CPU로 이동하고 NumPy 배열로 변환
            weights_np = self.current_weights.cpu().detach().numpy()
            
            # 시각화 모드 선택
            mode = self.mode_combo.currentText()
            
            # 필터 수 (최대 64개까지만 표시)
            num_filters = min(weights_np.shape[0], 64)
            
            # "푸리에 진폭 스펙트럼" 모드일 경우
            if mode == "푸리에 진폭 스펙트럼":
                # 모든 스펙트럼을 저장할 리스트
                all_spectrums = []
                all_pixmaps = []
                
                # 필터 갯수만큼 스펙트럼 계산
                for i in range(num_filters):
                    # 첫 번째 채널의 가중치를 사용하여 푸리에 변환
                    if weights_np.shape[1] > 0:
                        filter_img = weights_np[i, 0]
                        
                        # 커널 리사이즈 + 푸리에 변환 계산
                        magnitude = self.compute_fft_magnitude(filter_img)
                        
                        # 스펙트럼을 32x32로 리사이징
                        resized_spectrum = self.resize_spectrum_to_32x32(magnitude)
                        all_spectrums.append(resized_spectrum)
                        
                        # 진폭 스펙트럼 정규화 (0-255 범위로)
                        magnitude_normalized = ((resized_spectrum - resized_spectrum.min()) / 
                                              (resized_spectrum.max() - resized_spectrum.min() + 1e-8) * 255).astype(np.uint8)
                        
                        # 이미지 크기 조정 (32x32를 120x120으로 스케일링)
                        scale_factor = 120 // 32  # 3.75이지만 정수로 3
                        scaled_img = np.kron(magnitude_normalized, np.ones((scale_factor, scale_factor), dtype=np.uint8))
                        
                        # 정확한 크기로 맞추기 위해 추가 조정
                        if scaled_img.shape[0] < 120 or scaled_img.shape[1] < 120:
                            # 부족한 부분을 패딩으로 채움
                            pad_h = max(0, 120 - scaled_img.shape[0])
                            pad_w = max(0, 120 - scaled_img.shape[1])
                            scaled_img = np.pad(scaled_img, ((0, pad_h), (0, pad_w)), mode='edge')
                        
                        # 크기가 초과하는 경우 자르기
                        scaled_img = scaled_img[:120, :120]
                        
                        # QImage로 변환
                        q_img = QImage(scaled_img.data, scaled_img.shape[1], scaled_img.shape[0], 
                                    scaled_img.shape[1], QImage.Format.Format_Grayscale8)
                        pixmap = QPixmap.fromImage(q_img)
                        all_pixmaps.append(pixmap)
                    else:
                        # 채널이 없는 경우 빈 스펙트럼과 픽스맵
                        all_spectrums.append(np.zeros((32, 32)))
                        all_pixmaps.append(QPixmap(120, 120))
                
                # 코사인 유사도 기반 클러스터링 (임계값 0.7)
                if len(all_spectrums) > 1:
                    cluster_labels = self.compute_cosine_similarity_clustering(all_spectrums, threshold=0.93)
                    
                    # 클러스터 정보 출력
                    unique_clusters = np.unique(cluster_labels)
                    print(f"발견된 클러스터 수: {len(unique_clusters)}")
                    for cluster_id in unique_clusters:
                        if cluster_id != -1:  # -1은 노이즈(클러스터에 속하지 않음)
                            cluster_members = np.where(cluster_labels == cluster_id)[0]
                            print(f"클러스터 {cluster_id}: 필터 {cluster_members}")
                    
                    # 각 필터에 클러스터 색상으로 테두리 그리기
                    for i in range(num_filters):
                        pixmap = all_pixmaps[i]
                        cluster_id = cluster_labels[i]
                        
                        if cluster_id != -1:  # 클러스터에 속하는 경우
                            color_idx = cluster_id % len(self.cluster_colors)
                            border_color = self.cluster_colors[color_idx]
                            pixmap = self.draw_border_on_pixmap(pixmap, border_color)
                        
                        # 라벨 업데이트
                        if i < len(self.filter_labels):
                            self.filter_labels[i].setPixmap(pixmap)
                else:
                    # 스펙트럼이 1개 이하인 경우 클러스터링 없이 표시
                    for i in range(num_filters):
                        if i < len(self.filter_labels):
                            self.filter_labels[i].setPixmap(all_pixmaps[i])
                
                # 남은 라벨은 비움
                for i in range(num_filters, len(self.filter_labels)):
                    self.filter_labels[i].clear()
            
            # "컬러 채널 시각화" 모드일 경우 - RGB 채널을 모두 활용
            elif mode == "컬러 채널 시각화":
                # 필터 갯수만큼 표시
                for i in range(num_filters):
                    # 입력 채널이 3개 이상인 경우 RGB로 시각화
                    if weights_np.shape[1] >= 3:
                        # 필터의 모든 채널 사용 (R, G, B)
                        r_channel = weights_np[i, 0]  # R 채널
                        g_channel = weights_np[i, 1]  # G 채널
                        b_channel = weights_np[i, 2]  # B 채널
                    else:
                        # 입력 채널이 3개 미만인 경우, 흑백 시각화를 컬러로 변환
                        r_channel = weights_np[i, 0] if weights_np.shape[1] > 0 else np.zeros_like(weights_np[i, 0])
                        g_channel = weights_np[i, 0] if weights_np.shape[1] > 0 else np.zeros_like(weights_np[i, 0])
                        b_channel = weights_np[i, 0] if weights_np.shape[1] > 0 else np.zeros_like(weights_np[i, 0])
                    
                    # 각 채널 정규화
                    min_val = min(r_channel.min(), g_channel.min(), b_channel.min())
                    max_val = max(r_channel.max(), g_channel.max(), b_channel.max())
                    
                    # 값 범위 정규화 (0-1 사이로)
                    r_norm = np.clip((r_channel - min_val) / (max_val - min_val + 1e-8), 0, 1)
                    g_norm = np.clip((g_channel - min_val) / (max_val - min_val + 1e-8), 0, 1)
                    b_norm = np.clip((b_channel - min_val) / (max_val - min_val + 1e-8), 0, 1)
                    
                    # RGB 채널 결합
                    rgb_img = np.stack([r_norm, g_norm, b_norm], axis=2)
                    
                    # 이미지 스케일링
                    scale_factor = 10
                    h, w = rgb_img.shape[:2]
                    rgb_large = np.zeros((h*scale_factor, w*scale_factor, 3))
                    
                    # 단순 스케일링 (각 픽셀을 NxN 크기로 확대)
                    for y in range(h):
                        for x in range(w):
                            rgb_large[y*scale_factor:(y+1)*scale_factor, 
                                    x*scale_factor:(x+1)*scale_factor] = rgb_img[y, x]
                    
                    # RGB 이미지를 PyQt5 QImage로 변환
                    rgb_large = (rgb_large * 255).astype(np.uint8)
                    h, w = rgb_large.shape[:2]
                    
                    bytes_per_line = 3 * w
                    q_img = QImage(rgb_large.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)
                    
                    # 라벨 업데이트 (필터 수에 따라)
                    if i < len(self.filter_labels):
                        self.filter_labels[i].setPixmap(pixmap)
                
                # 남은 라벨은 비움
                for i in range(num_filters, len(self.filter_labels)):
                    self.filter_labels[i].clear()
            
            # "고대비 가중치" 모드일 경우 - 각 필터 내 값을 극대화하여 패턴 강조
            elif mode == "고대비 가중치":
                # 필터 갯수만큼 표시
                for i in range(num_filters):
                    # 각 필터의 첫 번째 채널 사용
                    filter_img = weights_np[i, 0] if weights_np.shape[1] > 0 else np.zeros((3, 3))
                    
                    # 히스토그램 평활화 적용
                    # NumPy의 histogram 함수를 사용하여 직접 구현
                    hist, bin_edges = np.histogram(filter_img.flatten(), bins=256)
                    cdf = hist.cumsum()
                    cdf_normalized = cdf * 255 / (cdf[-1] + 1e-8)  # 0-255 범위로 정규화
                    
                    # 이미지 매핑
                    shape = filter_img.shape
                    flat_img = filter_img.flatten()
                    bin_idx = np.digitize(flat_img, bin_edges[1:])
                    equalized_img = cdf_normalized[bin_idx].astype(np.uint8)
                    equalized_img = equalized_img.reshape(shape)
                    
                    # 이미지 스케일링
                    scale_factor = 10
                    scaled_img = np.kron(equalized_img, np.ones((scale_factor, scale_factor), dtype=np.uint8))
                    
                    q_img = QImage(scaled_img.data, scaled_img.shape[1], scaled_img.shape[0], 
                                scaled_img.shape[1], QImage.Format.Format_Grayscale8)
                    pixmap = QPixmap.fromImage(q_img)
                    
                    # 라벨 업데이트 (필터 수에 따라)
                    if i < len(self.filter_labels):
                        self.filter_labels[i].setPixmap(pixmap)
                
                # 남은 라벨은 비움
                for i in range(num_filters, len(self.filter_labels)):
                    self.filter_labels[i].clear()
            
            # "실제 가중치" 모드일 경우 - 원본 가중치 그대로 시각화
            else:
                # 전체 가중치의 범위 구하기
                if weights_np.shape[1] > 0:
                    all_weights_min = weights_np[:, 0].min()
                    all_weights_max = weights_np[:, 0].max()
                else:
                    all_weights_min, all_weights_max = 0, 1
                
                # 필터 갯수만큼 표시
                for i in range(num_filters):
                    # 첫 번째 채널만 사용하여 시각화
                    # 첫 번째 채널만 사용하여 시각화
                    filter_img = weights_np[i, 0] if weights_np.shape[1] > 0 else np.zeros((3, 3))
                   
                    normalized_img = ((filter_img - all_weights_min) / (all_weights_max - all_weights_min + 1e-8) * 255).astype(np.uint8)
                   
                    # 이미지 크기 조정
                    scale_factor = 10
                    scaled_img = np.kron(normalized_img, np.ones((scale_factor, scale_factor), dtype=np.uint8))
                   
                    q_img = QImage(scaled_img.data, scaled_img.shape[1], scaled_img.shape[0], 
                               scaled_img.shape[1], QImage.Format.Format_Grayscale8)
                    pixmap = QPixmap.fromImage(q_img)
                   
                    # 라벨 업데이트 (필터 수에 따라)
                    if i < len(self.filter_labels):
                        self.filter_labels[i].setPixmap(pixmap)
                   
                # 남은 라벨은 비움
                for i in range(num_filters, len(self.filter_labels)):
                    self.filter_labels[i].clear()
   
    # PyQt 애플리케이션 초기화
    app = QApplication(sys.argv)
    window = WeightVisualizerWindow()
    window.show()
   
    # 애플리케이션 실행
    sys.exit(app.exec_())
