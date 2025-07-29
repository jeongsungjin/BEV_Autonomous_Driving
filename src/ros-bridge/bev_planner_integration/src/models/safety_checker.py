#!/usr/bin/env python3
"""
안전성 검사 및 손실 함수 모듈
BEV-Planner의 collision loss와 lane keeping 로직을 단순화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class SafetyChecker:
    """
    궤적의 안전성을 검사하고 위험도를 계산하는 클래스
    """
    
    def __init__(self,
                 bev_height: int = 48,
                 bev_width: int = 80,
                 ego_vehicle_size: Tuple[float, float] = (2.0, 4.5),  # (width, length) in meters
                 safety_margin: float = 1.0):
        
        self.bev_height = bev_height
        self.bev_width = bev_width
        self.ego_width, self.ego_length = ego_vehicle_size
        self.safety_margin = safety_margin
        
        # BEV 좌표계 설정 (중앙이 ego vehicle 위치)
        self.bev_center_x = bev_width // 2
        self.bev_center_y = bev_height // 2
        
        # 실제 거리 스케일 (BEV 전체가 대략 40m x 30m를 커버한다고 가정)
        self.meters_per_pixel_x = 40.0 / bev_width  # ~0.5m/pixel
        self.meters_per_pixel_y = 30.0 / bev_height  # ~0.625m/pixel
        
    def check_collision_risk(self,
                           trajectory: torch.Tensor,
                           detection_mask: torch.Tensor) -> torch.Tensor:
        """
        궤적과 장애물 간의 충돌 위험도 계산
        
        Args:
            trajectory: [B, num_steps, 2] 궤적 (meter 단위)
            detection_mask: [B, H, W] 객체 검출 마스크
            
        Returns:
            collision_risk: [B, num_steps] 각 스텝의 충돌 위험도 (0~1)
        """
        batch_size, num_steps, _ = trajectory.shape
        collision_risks = []
        
        for step in range(num_steps):
            step_pos = trajectory[:, step, :]  # [B, 2]
            
            # 미터 단위 좌표를 픽셀 좌표로 변환
            pixel_x = (step_pos[:, 0] / self.meters_per_pixel_x + self.bev_center_x).long()
            pixel_y = (step_pos[:, 1] / self.meters_per_pixel_y + self.bev_center_y).long()
            
            # BEV 범위 내로 클램핑
            pixel_x = torch.clamp(pixel_x, 0, self.bev_width - 1)
            pixel_y = torch.clamp(pixel_y, 0, self.bev_height - 1)
            
            # Ego vehicle 크기를 고려한 occupancy 확인
            step_risk = torch.zeros(batch_size, device=trajectory.device)
            
            for b in range(batch_size):
                x, y = pixel_x[b].item(), pixel_y[b].item()
                
                # Ego vehicle footprint 계산 (안전 여유거리 포함)
                ego_half_width = int((self.ego_width / 2 + self.safety_margin) / self.meters_per_pixel_x)
                ego_half_length = int((self.ego_length / 2 + self.safety_margin) / self.meters_per_pixel_y)
                
                x_min = max(0, x - ego_half_width)
                x_max = min(self.bev_width, x + ego_half_width + 1)
                y_min = max(0, y - ego_half_length)
                y_max = min(self.bev_height, y + ego_half_length + 1)
                
                # 해당 영역의 검출 마스크 확인
                footprint_detections = detection_mask[b, y_min:y_max, x_min:x_max]
                step_risk[b] = footprint_detections.float().max()  # 최대값 = 충돌 위험도
            
            collision_risks.append(step_risk)
        
        return torch.stack(collision_risks, dim=1)  # [B, num_steps]
    
    def check_lane_compliance(self,
                            trajectory: torch.Tensor,
                            drivable_area_mask: torch.Tensor) -> torch.Tensor:
        """
        궤적이 주행 가능 영역 내에 있는지 확인
        
        Args:
            trajectory: [B, num_steps, 2] 궤적
            drivable_area_mask: [B, H, W] 주행 가능 영역 마스크
            
        Returns:
            lane_compliance: [B, num_steps] 차선 준수도 (0~1, 1이 완전 준수)
        """
        batch_size, num_steps, _ = trajectory.shape
        compliance_scores = []
        
        for step in range(num_steps):
            step_pos = trajectory[:, step, :]  # [B, 2]
            
            # 픽셀 좌표 변환
            pixel_x = (step_pos[:, 0] / self.meters_per_pixel_x + self.bev_center_x).long()
            pixel_y = (step_pos[:, 1] / self.meters_per_pixel_y + self.bev_center_y).long()
            
            pixel_x = torch.clamp(pixel_x, 0, self.bev_width - 1)
            pixel_y = torch.clamp(pixel_y, 0, self.bev_height - 1)
            
            step_compliance = torch.zeros(batch_size, device=trajectory.device)
            
            for b in range(batch_size):
                x, y = pixel_x[b].item(), pixel_y[b].item()
                
                # Ego vehicle footprint 영역의 주행 가능 영역 비율 계산
                ego_half_width = int(self.ego_width / 2 / self.meters_per_pixel_x)
                ego_half_length = int(self.ego_length / 2 / self.meters_per_pixel_y)
                
                x_min = max(0, x - ego_half_width)
                x_max = min(self.bev_width, x + ego_half_width + 1)
                y_min = max(0, y - ego_half_length)
                y_max = min(self.bev_height, y + ego_half_length + 1)
                
                footprint_drivable = drivable_area_mask[b, y_min:y_max, x_min:x_max]
                step_compliance[b] = footprint_drivable.float().mean()  # 평균 = 준수도
            
            compliance_scores.append(step_compliance)
        
        return torch.stack(compliance_scores, dim=1)  # [B, num_steps]


class PlanningLoss(nn.Module):
    """
    BEV-Planner 스타일의 경로 계획 손실 함수
    """
    
    def __init__(self,
                 trajectory_weight: float = 1.0,
                 collision_weight: float = 5.0,
                 lane_keeping_weight: float = 2.0,
                 smoothness_weight: float = 0.5,
                 confidence_weight: float = 0.1):
        
        super(PlanningLoss, self).__init__()
        
        self.trajectory_weight = trajectory_weight
        self.collision_weight = collision_weight  
        self.lane_keeping_weight = lane_keeping_weight
        self.smoothness_weight = smoothness_weight
        self.confidence_weight = confidence_weight
        
        self.safety_checker = SafetyChecker()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
    def forward(self,
                predicted_trajectory: torch.Tensor,
                predicted_confidence: torch.Tensor,
                target_trajectory: Optional[torch.Tensor] = None,
                detection_mask: Optional[torch.Tensor] = None,
                drivable_area_mask: Optional[torch.Tensor] = None,
                lane_line_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        종합적인 계획 손실 계산
        
        Args:
            predicted_trajectory: [B, num_steps, 2] 예측 궤적
            predicted_confidence: [B, num_steps] 예측 신뢰도
            target_trajectory: [B, num_steps, 2] 타겟 궤적 (optional)
            detection_mask: [B, H, W] 객체 검출 마스크
            drivable_area_mask: [B, H, W] 주행 가능 영역 마스크
            lane_line_mask: [B, H, W] 차선 마스크
            
        Returns:
            Dict of losses
        """
        losses = {}
        total_loss = 0.0
        
        # 1. 궤적 정확도 손실 (supervised learning의 경우)
        if target_trajectory is not None:
            trajectory_loss = self.mse_loss(predicted_trajectory, target_trajectory)
            losses['trajectory_loss'] = trajectory_loss
            total_loss += self.trajectory_weight * trajectory_loss
        
        # 2. 충돌 회피 손실
        if detection_mask is not None:
            collision_risks = self.safety_checker.check_collision_risk(
                predicted_trajectory, detection_mask
            )
            # 충돌 위험이 높을수록 큰 손실
            collision_loss = collision_risks.mean()
            losses['collision_loss'] = collision_loss
            total_loss += self.collision_weight * collision_loss
        
        # 3. 차선 유지 손실
        if drivable_area_mask is not None:
            lane_compliance = self.safety_checker.check_lane_compliance(
                predicted_trajectory, drivable_area_mask
            )
            # 차선 준수도가 낮을수록 큰 손실 (1 - compliance)
            lane_keeping_loss = (1.0 - lane_compliance).mean()
            losses['lane_keeping_loss'] = lane_keeping_loss
            total_loss += self.lane_keeping_weight * lane_keeping_loss
        
        # 4. 궤적 부드러움 손실
        if predicted_trajectory.size(1) > 1:
            # 연속된 스텝 간의 가속도 변화를 최소화
            velocity = predicted_trajectory[:, 1:] - predicted_trajectory[:, :-1]  # [B, num_steps-1, 2]
            if velocity.size(1) > 1:
                acceleration = velocity[:, 1:] - velocity[:, :-1]  # [B, num_steps-2, 2]
                smoothness_loss = torch.norm(acceleration, dim=-1).mean()
                losses['smoothness_loss'] = smoothness_loss
                total_loss += self.smoothness_weight * smoothness_loss
        
        # 5. 신뢰도 일관성 손실
        if predicted_confidence is not None:
            # 높은 위험 상황에서는 낮은 신뢰도를 가져야 함
            if detection_mask is not None:
                collision_risks = self.safety_checker.check_collision_risk(
                    predicted_trajectory, detection_mask
                )
                # 충돌 위험이 높으면 신뢰도는 낮아야 함
                expected_confidence = 1.0 - collision_risks
                confidence_loss = self.mse_loss(predicted_confidence, expected_confidence)
                losses['confidence_loss'] = confidence_loss
                total_loss += self.confidence_weight * confidence_loss
        
        losses['total_loss'] = total_loss
        return losses
    
    def compute_metrics(self,
                       predicted_trajectory: torch.Tensor,
                       target_trajectory: Optional[torch.Tensor] = None,
                       detection_mask: Optional[torch.Tensor] = None,
                       drivable_area_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        평가 메트릭 계산
        
        Returns:
            Dict of metrics
        """
        metrics = {}
        
        # 궤적 정확도 (ADE - Average Displacement Error)
        if target_trajectory is not None:
            displacement_errors = torch.norm(
                predicted_trajectory - target_trajectory, dim=-1
            )  # [B, num_steps]
            ade = displacement_errors.mean().item()
            metrics['ADE'] = ade
            
            # FDE (Final Displacement Error)
            fde = displacement_errors[:, -1].mean().item()
            metrics['FDE'] = fde
        
        # 충돌률
        if detection_mask is not None:
            collision_risks = self.safety_checker.check_collision_risk(
                predicted_trajectory, detection_mask
            )
            collision_rate = (collision_risks > 0.5).float().mean().item()
            metrics['collision_rate'] = collision_rate
        
        # 차선 준수율
        if drivable_area_mask is not None:
            lane_compliance = self.safety_checker.check_lane_compliance(
                predicted_trajectory, drivable_area_mask
            )
            lane_compliance_rate = (lane_compliance > 0.8).float().mean().item()
            metrics['lane_compliance_rate'] = lane_compliance_rate
        
        return metrics


if __name__ == "__main__":
    # 테스트
    print("🧪 SafetyChecker 및 PlanningLoss 테스트...")
    
    # 테스트 데이터
    batch_size = 2
    num_steps = 6
    bev_h, bev_w = 48, 80
    
    predicted_traj = torch.randn(batch_size, num_steps, 2) * 5  # ±5m 범위
    predicted_conf = torch.rand(batch_size, num_steps)
    target_traj = torch.randn(batch_size, num_steps, 2) * 5
    
    # 가짜 BEV 마스크들
    det_mask = torch.rand(batch_size, bev_h, bev_w) > 0.8  # 20% 객체
    da_mask = torch.rand(batch_size, bev_h, bev_w) > 0.3   # 70% 주행가능
    ll_mask = torch.rand(batch_size, bev_h, bev_w) > 0.9   # 10% 차선
    
    # 손실 함수 테스트
    loss_fn = PlanningLoss()
    losses = loss_fn(
        predicted_traj, predicted_conf, target_traj,
        det_mask.float(), da_mask.float(), ll_mask.float()
    )
    
    print("✅ 손실 함수 테스트 완료!")
    for name, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"   - {name}: {value.item():.4f}")
    
    # 메트릭 테스트  
    metrics = loss_fn.compute_metrics(
        predicted_traj, target_traj, det_mask.float(), da_mask.float()
    )
    
    print("✅ 메트릭 계산 완료!")
    for name, value in metrics.items():
        print(f"   - {name}: {value:.4f}") 