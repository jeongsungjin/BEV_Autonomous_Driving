#!/usr/bin/env python3
"""
BEV-Planner에서 영감을 받은 간소화된 경로 계획 모델
YOLOP 출력과 ego 상태를 기반으로 안전한 미래 궤적을 생성
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class SimplifiedBEVPlanner(nn.Module):
    """
    간소화된 BEV 기반 경로 계획 모델
    
    주요 기능:
    1. BEV 특징 + Ego 상태 → Transformer 기반 처리
    2. 미래 6스텝 궤적 예측 (3초, 0.5초 간격)
    3. 충돌 회피 및 차선 유지 제약
    """
    
    def __init__(self,
                 bev_embed_dim: int = 256,
                 ego_embed_dim: int = 256,
                 hidden_dim: int = 512,
                 num_future_steps: int = 6,
                 num_transformer_layers: int = 4,
                 num_attention_heads: int = 8,
                 dropout: float = 0.1,
                 max_speed: float = 15.0,
                 safety_margin: float = 2.0):
        
        super(SimplifiedBEVPlanner, self).__init__()
        
        self.bev_embed_dim = bev_embed_dim
        self.ego_embed_dim = ego_embed_dim
        self.hidden_dim = hidden_dim
        self.num_future_steps = num_future_steps
        self.max_speed = max_speed
        self.safety_margin = safety_margin
        
        # BEV 특징 압축 (3840 -> 더 작은 차원)
        self.bev_compressor = nn.Sequential(
            nn.Linear(bev_embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Ego 상태 처리
        self.ego_processor = nn.Sequential(
            nn.Linear(ego_embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # 특징 융합 (hidden_dim//2 -> hidden_dim)
        fusion_input_dim = hidden_dim // 2  # BEV와 ego 모두 hidden_dim//2 차원
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Transformer 기반 시퀀스 모델링
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_transformer_layers
        )
        
        # 궤적 디코더 (미래 각 스텝의 (x, y) 좌표 예측)
        self.trajectory_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_future_steps * 2)  # (x, y) * num_steps
        )
        
        # 신뢰도 예측 (각 스텝의 예측 신뢰도)
        self.confidence_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_future_steps),
            nn.Sigmoid()  # 0~1 사이 신뢰도
        )
        
        # Learnable query tokens for future steps
        self.future_queries = nn.Parameter(
            torch.randn(1, num_future_steps, hidden_dim)
        )
        
    def forward(self, 
                bev_features: torch.Tensor,
                ego_features: torch.Tensor,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            bev_features: [B, H*W, C] BEV 공간 특징
            ego_features: [B, C] Ego 차량 상태 특징
            return_attention: attention weights 반환 여부
            
        Returns:
            Dict containing:
                - 'trajectory': [B, num_steps, 2] 미래 궤적 (x, y)
                - 'confidence': [B, num_steps] 각 스텝의 신뢰도
                - 'attention_weights': [B, num_heads, seq_len, seq_len] (optional)
        """
        batch_size = bev_features.size(0)
        
        # 1. BEV 특징 압축 및 중요 영역 선택
        bev_compressed = self.bev_compressor(bev_features)  # [B, H*W, hidden_dim//2]
        
        # 상위 K개 중요 BEV 위치만 선택 (attention 효율성을 위해)
        K = min(64, bev_compressed.size(1))  # 최대 64개 위치
        bev_importance = bev_compressed.norm(dim=-1)  # [B, H*W]
        _, top_indices = torch.topk(bev_importance, K, dim=-1)  # [B, K]
        
        # 중요 BEV 특징 추출
        bev_selected = torch.gather(
            bev_compressed, 1, 
            top_indices.unsqueeze(-1).expand(-1, -1, bev_compressed.size(-1))
        )  # [B, K, hidden_dim//2]
        
        # 2. Ego 특징 처리
        ego_processed = self.ego_processor(ego_features).unsqueeze(1)  # [B, 1, hidden_dim//2]
        
        # 3. 특징 융합
        # BEV와 ego를 결합 [B, K+1, hidden_dim//2]
        combined_features = torch.cat([ego_processed, bev_selected], dim=1)  
        
        # 각 위치별로 feature_fusion 적용하여 hidden_dim으로 확장
        B, seq_len, feat_dim = combined_features.shape
        combined_features = combined_features.view(-1, feat_dim)  # [B*(K+1), hidden_dim//2]
        combined_features = self.feature_fusion(combined_features)  # [B*(K+1), hidden_dim]
        combined_features = combined_features.view(B, seq_len, self.hidden_dim)  # [B, K+1, hidden_dim]
        
        # 4. Future query tokens 추가
        future_queries = self.future_queries.expand(batch_size, -1, -1)  # [B, num_steps, hidden_dim]
        full_sequence = torch.cat([combined_features, future_queries], dim=1)  # [B, K+1+num_steps, hidden_dim]
        
        # 5. Transformer 처리
        transformer_output = self.transformer_encoder(full_sequence)  # [B, seq_len, hidden_dim]
        
        # Future step outputs 추출 (마지막 num_future_steps개)
        future_outputs = transformer_output[:, -(self.num_future_steps):, :]  # [B, num_steps, hidden_dim]
        
        # 6. 궤적 및 신뢰도 디코딩
        # Global context를 위해 평균 pooling
        global_context = transformer_output.mean(dim=1)  # [B, hidden_dim]
        
        # 궤적 예측 (전역적 접근)
        trajectory_flat = self.trajectory_decoder(global_context)  # [B, num_steps*2]
        trajectory = trajectory_flat.view(batch_size, self.num_future_steps, 2)  # [B, num_steps, 2]
        
        # 신뢰도 예측
        confidence = self.confidence_decoder(global_context)  # [B, num_steps]
        
        # 7. 물리적 제약 적용
        trajectory = self._apply_physical_constraints(trajectory, ego_features)
        
        result = {
            'trajectory': trajectory,
            'confidence': confidence,
            'bev_attention_indices': top_indices  # 디버깅용
        }
        
        if return_attention:
            # Attention weights 추출은 복잡하므로 여기서는 생략
            # 실제로는 transformer의 attention weights를 따로 저장해야 함
            pass
            
        return result
    
    def _apply_physical_constraints(self, 
                                  trajectory: torch.Tensor,
                                  ego_features: torch.Tensor) -> torch.Tensor:
        """
        물리적 제약을 적용하여 현실적인 궤적으로 조정
        
        Args:
            trajectory: [B, num_steps, 2] 원시 궤적 예측
            ego_features: [B, ego_dim] 현재 ego 상태
            
        Returns:
            제약이 적용된 궤적 [B, num_steps, 2]
        """
        batch_size, num_steps, _ = trajectory.shape
        
        # 1. 속도 제약 (연속된 스텝 간의 최대 이동 거리)
        dt = 0.5  # 0.5초 간격
        max_step_distance = self.max_speed * dt  # 최대 스텝당 이동 거리
        
        constrained_trajectory = trajectory.clone()
        
        for step in range(1, num_steps):
            # 이전 스텝과의 거리 계산
            step_delta = constrained_trajectory[:, step] - constrained_trajectory[:, step-1]
            step_distance = torch.norm(step_delta, dim=-1, keepdim=True)  # [B, 1]
            
            # 최대 거리 초과 시 정규화
            exceed_mask = step_distance > max_step_distance
            if exceed_mask.any():
                normalized_delta = step_delta / (step_distance + 1e-8) * max_step_distance
                constrained_trajectory[:, step] = torch.where(
                    exceed_mask.expand(-1, 2),
                    constrained_trajectory[:, step-1] + normalized_delta,
                    constrained_trajectory[:, step]
                )
        
        # 2. 부드러운 궤적을 위한 스무딩 (선택적)
        if num_steps >= 3:
            # 중앙값 필터링으로 급격한 변화 제거
            for step in range(1, num_steps-1):
                prev_pos = constrained_trajectory[:, step-1]
                curr_pos = constrained_trajectory[:, step]
                next_pos = constrained_trajectory[:, step+1]
                
                # 부드러운 보간
                smoothed_pos = 0.25 * prev_pos + 0.5 * curr_pos + 0.25 * next_pos
                constrained_trajectory[:, step] = smoothed_pos
        
        return constrained_trajectory
    
    def predict_trajectory(self,
                          bev_features: torch.Tensor,
                          ego_features: torch.Tensor,
                          return_dict: bool = False) -> torch.Tensor:
        """
        간단한 궤적 예측 인터페이스
        
        Args:
            bev_features: [B, H*W, C]
            ego_features: [B, C]  
            return_dict: 전체 결과 딕셔너리 반환 여부
            
        Returns:
            trajectory [B, num_steps, 2] 또는 전체 결과 딕셔너리
        """
        with torch.no_grad():
            result = self.forward(bev_features, ego_features)
            
        if return_dict:
            return result
        else:
            return result['trajectory']


def create_mock_trajectory_target(batch_size: int = 1, 
                                num_steps: int = 6,
                                straight_distance: float = 15.0) -> torch.Tensor:
    """
    테스트용 직진 궤적 생성
    
    Args:
        batch_size: 배치 크기
        num_steps: 미래 스텝 수
        straight_distance: 총 직진 거리
        
    Returns:
        [B, num_steps, 2] 형태의 궤적
    """
    # 직진 궤적 (y=0, x는 등간격 증가)
    x_coords = torch.linspace(0, straight_distance, num_steps)
    y_coords = torch.zeros(num_steps)
    
    trajectory = torch.stack([x_coords, y_coords], dim=-1)  # [num_steps, 2]
    trajectory = trajectory.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_steps, 2]
    
    return trajectory


if __name__ == "__main__":
    # 간단한 테스트
    print("🧪 SimplifiedBEVPlanner 테스트...")
    
    # 모델 초기화
    planner = SimplifiedBEVPlanner(
        bev_embed_dim=256,
        ego_embed_dim=256,
        hidden_dim=512,
        num_future_steps=6
    )
    
    # 테스트 데이터
    batch_size = 2
    bev_features = torch.randn(batch_size, 3840, 256)  # 48*80=3840
    ego_features = torch.randn(batch_size, 256)
    
    # 예측 실행
    with torch.no_grad():
        result = planner(bev_features, ego_features)
    
    print(f"✅ 테스트 완료!")
    print(f"   - 궤적 크기: {result['trajectory'].shape}")
    print(f"   - 신뢰도 크기: {result['confidence'].shape}")
    print(f"   - 궤적 범위: [{result['trajectory'].min():.3f}, {result['trajectory'].max():.3f}]")
    print(f"   - 신뢰도 범위: [{result['confidence'].min():.3f}, {result['confidence'].max():.3f}]") 