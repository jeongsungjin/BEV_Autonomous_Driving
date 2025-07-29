#!/usr/bin/env python3
"""
YOLOP 출력을 BEV-Planner 입력으로 변환하는 어댑터
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Tuple, Optional


class YOLOPToBEVAdapter(nn.Module):
    """
    YOLOP의 3개 마스크 출력을 BEV-Planner 형식의 특징으로 변환
    
    입력:
        - detection_mask: [H, W] 객체 검출 마스크
        - drivable_area_mask: [H, W] 주행 가능 영역 마스크  
        - lane_line_mask: [H, W] 차선 마스크
        
    출력:
        - bev_features: [B, H*W, C] BEV-Planner 형식 특징
    """
    
    def __init__(self, 
                 input_height: int = 48,
                 input_width: int = 80, 
                 embed_dim: int = 256,
                 use_positional_encoding: bool = True):
        super(YOLOPToBEVAdapter, self).__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        self.embed_dim = embed_dim
        self.use_positional_encoding = use_positional_encoding
        
        # 각 마스크를 독립적으로 임베딩
        self.det_embedding = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((input_height, input_width))
        )
        
        self.da_embedding = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((input_height, input_width))
        )
        
        self.ll_embedding = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((input_height, input_width))
        )
        
        # 특징 융합 및 최종 임베딩
        fusion_input_dim = 128 * 3  # det + da + ll
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(fusion_input_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        if self.use_positional_encoding:
            self.pos_encoding = self._create_positional_encoding()
            
        # ego 상태 임베딩 (속도, 조향각 등)
        self.ego_embedding = nn.Sequential(
            nn.Linear(6, 128),  # [vx, vy, v_norm, steering, yaw_rate, acceleration]
            nn.ReLU(inplace=True),
            nn.Linear(128, embed_dim),
            nn.ReLU(inplace=True)
        )
        
    def _create_positional_encoding(self) -> torch.Tensor:
        """BEV 그리드를 위한 2D positional encoding 생성"""
        pos_h = torch.arange(self.input_height).float()
        pos_w = torch.arange(self.input_width).float()
        
        # 정규화 (-1 ~ 1)
        pos_h = (pos_h / (self.input_height - 1)) * 2 - 1
        pos_w = (pos_w / (self.input_width - 1)) * 2 - 1
        
        # 메쉬그리드 생성
        grid_h, grid_w = torch.meshgrid(pos_h, pos_w, indexing='ij')
        
        # Sinusoidal encoding
        pe = torch.zeros(self.embed_dim, self.input_height, self.input_width)
        
        # 임베딩 차원을 4로 나누어 각 방향과 sin/cos를 위한 차원 수 계산
        d_model = self.embed_dim // 4
        div_term = torch.exp(torch.arange(0, d_model, 1).float() * 
                           -(np.log(10000.0) / d_model))
        
        # H 방향 encoding (sin/cos)
        pe[0:d_model] = torch.sin(grid_h.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))
        pe[d_model:2*d_model] = torch.cos(grid_h.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))
        
        # W 방향 encoding (sin/cos)
        pe[2*d_model:3*d_model] = torch.sin(grid_w.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))
        pe[3*d_model:4*d_model] = torch.cos(grid_w.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))
        
        return pe
        
    def forward(self, 
                detection_mask: torch.Tensor,
                drivable_area_mask: torch.Tensor, 
                lane_line_mask: torch.Tensor,
                ego_status: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            detection_mask: [B, H, W] 또는 [H, W]
            drivable_area_mask: [B, H, W] 또는 [H, W]  
            lane_line_mask: [B, H, W] 또는 [H, W]
            ego_status: 딕셔너리 {'velocity': [vx, vy], 'steering': angle, 'yaw_rate': rate}
            
        Returns:
            Dict with 'bev_features': [B, H*W, C], 'ego_features': [B, C]
        """
        # 배치 차원 확인 및 추가
        if detection_mask.dim() == 2:
            detection_mask = detection_mask.unsqueeze(0)
        if drivable_area_mask.dim() == 2:
            drivable_area_mask = drivable_area_mask.unsqueeze(0)
        if lane_line_mask.dim() == 2:
            lane_line_mask = lane_line_mask.unsqueeze(0)
            
        batch_size = detection_mask.size(0)
        
        # 채널 차원 추가 [B, 1, H, W]
        det_input = detection_mask.unsqueeze(1).float()
        da_input = drivable_area_mask.unsqueeze(1).float()
        ll_input = lane_line_mask.unsqueeze(1).float()
        
        # 각 마스크 임베딩
        det_feat = self.det_embedding(det_input)    # [B, 128, H, W]
        da_feat = self.da_embedding(da_input)       # [B, 128, H, W]
        ll_feat = self.ll_embedding(ll_input)       # [B, 128, H, W]
        
        # 특징 결합 [B, 384, H, W]
        combined_feat = torch.cat([det_feat, da_feat, ll_feat], dim=1)
        
        # 최종 임베딩 [B, embed_dim, H, W]
        bev_feat = self.feature_fusion(combined_feat)
        
        # Positional encoding 추가
        if self.use_positional_encoding:
            pos_enc = self.pos_encoding.to(bev_feat.device).unsqueeze(0)
            bev_feat = bev_feat + pos_enc
            
        # BEV-Planner 형식으로 변환 [B, H*W, C]
        B, C, H, W = bev_feat.shape
        bev_features = bev_feat.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        result = {'bev_features': bev_features}
        
        # Ego 상태 처리
        if ego_status is not None:
            ego_vector = self._process_ego_status(ego_status, batch_size, bev_feat.device)
            ego_features = self.ego_embedding(ego_vector)  # [B, embed_dim]
            result['ego_features'] = ego_features
            
        return result
    
    def _process_ego_status(self, ego_status: Dict[str, float], 
                          batch_size: int, device: torch.device) -> torch.Tensor:
        """Ego 상태를 벡터로 변환"""
        # 기본값 설정
        velocity = ego_status.get('velocity', [0.0, 0.0])
        steering = ego_status.get('steering', 0.0)
        yaw_rate = ego_status.get('yaw_rate', 0.0) 
        acceleration = ego_status.get('acceleration', 0.0)
        
        # 속도 크기 계산
        v_norm = np.sqrt(velocity[0]**2 + velocity[1]**2)
        
        # 6차원 벡터 생성 [vx, vy, v_norm, steering, yaw_rate, acceleration]
        ego_vector = torch.tensor([
            velocity[0], velocity[1], v_norm, 
            steering, yaw_rate, acceleration
        ], device=device).float()
        
        # 배치 차원 추가
        ego_vector = ego_vector.unsqueeze(0).repeat(batch_size, 1)
        
        return ego_vector


class BEVFeatureProcessor:
    """
    ROS OccupancyGrid 메시지를 PyTorch 텐서로 변환하는 유틸리티
    """
    
    @staticmethod
    def occupancy_grid_to_tensor(grid_msg, target_height: int = 48, 
                               target_width: int = 80) -> torch.Tensor:
        """
        OccupancyGrid 메시지를 PyTorch 텐서로 변환
        
        Args:
            grid_msg: nav_msgs/OccupancyGrid
            target_height: 목표 높이
            target_width: 목표 너비
            
        Returns:
            torch.Tensor: [H, W] 형태의 텐서
        """
        # OccupancyGrid 데이터 추출
        width = grid_msg.info.width
        height = grid_msg.info.height
        data = np.array(grid_msg.data, dtype=np.float32)
        
        # 2D 배열로 변환
        grid_2d = data.reshape(height, width)
        
        # 값 정규화 (-1~100 → 0~1)
        grid_2d = np.clip(grid_2d, 0, 100) / 100.0
        
        # 목표 크기로 리사이즈
        if height != target_height or width != target_width:
            grid_2d = cv2.resize(grid_2d, (target_width, target_height), 
                               interpolation=cv2.INTER_LINEAR)
        
        return torch.from_numpy(grid_2d)
    
    @staticmethod  
    def create_mock_ego_status(velocity_x: float = 5.0, 
                             velocity_y: float = 0.0,
                             steering: float = 0.0) -> Dict[str, float]:
        """
        테스트용 mock ego 상태 생성
        
        Args:
            velocity_x: 전진 속도 (m/s)
            velocity_y: 측면 속도 (m/s) 
            steering: 조향각 (rad)
            
        Returns:
            ego 상태 딕셔너리
        """
        return {
            'velocity': [velocity_x, velocity_y],
            'steering': steering,
            'yaw_rate': steering * 0.1,  # 간단한 추정
            'acceleration': 0.0
        } 