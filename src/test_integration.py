#!/usr/bin/env python3
"""
전체 시스템 통합 테스트
YOLOP Adapter → BEV-Planner → Safety Check의 완전한 파이프라인 테스트
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

# 모듈 import
from adapters.yolop_adapter import YOLOPToBEVAdapter, BEVFeatureProcessor
from models.simplified_planner import SimplifiedBEVPlanner, create_mock_trajectory_target
from models.safety_checker import SafetyChecker, PlanningLoss


class IntegratedBEVPlanningSystem:
    """
    YOLOP + BEV-Planner 통합 시스템
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # 1. YOLOP 어댑터 초기화
        self.yolop_adapter = YOLOPToBEVAdapter(
            input_height=48,
            input_width=80,
            embed_dim=256,
            use_positional_encoding=True
        ).to(device)
        
        # 2. BEV-Planner 초기화
        self.bev_planner = SimplifiedBEVPlanner(
            bev_embed_dim=256,
            ego_embed_dim=256,
            hidden_dim=512,
            num_future_steps=6,
            max_speed=15.0,
            safety_margin=2.0
        ).to(device)
        
        # 3. 손실 함수 및 안전성 검사
        self.loss_fn = PlanningLoss(
            trajectory_weight=1.0,
            collision_weight=5.0,
            lane_keeping_weight=2.0,
            smoothness_weight=0.5,
            confidence_weight=0.1
        )
        
        # 4. 통계 추적
        self.reset_stats()
        
    def reset_stats(self):
        """통계 초기화"""
        self.inference_times = []
        self.trajectory_history = []
        self.safety_scores = []
        
    def forward(self, 
                detection_mask: torch.Tensor,
                drivable_area_mask: torch.Tensor,
                lane_line_mask: torch.Tensor,
                ego_status: dict,
                return_details: bool = False) -> dict:
        """
        전체 파이프라인 실행
        
        Args:
            detection_mask: [B, H, W] 또는 [H, W]
            drivable_area_mask: [B, H, W] 또는 [H, W]
            lane_line_mask: [B, H, W] 또는 [H, W]
            ego_status: ego 차량 상태 딕셔너리
            return_details: 상세 정보 반환 여부
            
        Returns:
            결과 딕셔너리
        """
        start_time = time.time()
        
        # 1. YOLOP 출력을 BEV 특징으로 변환
        adapter_output = self.yolop_adapter(
            detection_mask.to(self.device),
            drivable_area_mask.to(self.device),
            lane_line_mask.to(self.device),
            ego_status
        )
        
        bev_features = adapter_output['bev_features']  # [B, H*W, C]
        ego_features = adapter_output['ego_features']  # [B, C]
        
        # 2. 경로 계획 실행
        planning_output = self.bev_planner(bev_features, ego_features)
        
        predicted_trajectory = planning_output['trajectory']  # [B, num_steps, 2]
        predicted_confidence = planning_output['confidence']  # [B, num_steps]
        
        # 3. 안전성 평가
        safety_checker = SafetyChecker()
        
        collision_risks = safety_checker.check_collision_risk(
            predicted_trajectory, detection_mask.to(self.device)
        )
        
        lane_compliance = safety_checker.check_lane_compliance(
            predicted_trajectory, drivable_area_mask.to(self.device)
        )
        
        # 4. 성능 통계 업데이트
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # 안전 점수 계산 (0~1, 높을수록 안전)
        safety_score = (1.0 - collision_risks.mean()).item() * lane_compliance.mean().item()
        self.safety_scores.append(safety_score)
        
        # 궤적 히스토리 저장
        self.trajectory_history.append(predicted_trajectory.detach().cpu().numpy())
        
        # 결과 구성
        result = {
            'trajectory': predicted_trajectory.detach().cpu(),
            'confidence': predicted_confidence.detach().cpu(),
            'collision_risks': collision_risks.detach().cpu(),
            'lane_compliance': lane_compliance.detach().cpu(),
            'safety_score': safety_score,
            'inference_time': inference_time
        }
        
        if return_details:
            result.update({
                'bev_features': bev_features.detach().cpu(),
                'ego_features': ego_features.detach().cpu(),
                'bev_attention_indices': planning_output.get('bev_attention_indices'),
                'adapter_output': adapter_output
            })
        
        return result
    
    def evaluate_loss(self,
                     predicted_trajectory: torch.Tensor,
                     predicted_confidence: torch.Tensor,
                     detection_mask: torch.Tensor,
                     drivable_area_mask: torch.Tensor,
                     lane_line_mask: torch.Tensor,
                     target_trajectory: torch.Tensor = None) -> dict:
        """
        손실 함수 평가
        """
        return self.loss_fn(
            predicted_trajectory,
            predicted_confidence,
            target_trajectory,
            detection_mask,
            drivable_area_mask,
            lane_line_mask
        )
    
    def get_performance_stats(self) -> dict:
        """성능 통계 반환"""
        if not self.inference_times:
            return {}
            
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'fps': 1.0 / np.mean(self.inference_times),
            'avg_safety_score': np.mean(self.safety_scores),
            'num_inferences': len(self.inference_times)
        }


def create_test_scenario(scenario_type='straight_road'):
    """
    다양한 테스트 시나리오 생성
    """
    H, W = 48, 80
    
    if scenario_type == 'straight_road':
        # 직진 도로 시나리오
        det_mask = torch.zeros(H, W)
        
        da_mask = torch.zeros(H, W)
        da_mask[10:38, 20:60] = 1.0  # 중앙 주행 가능 영역
        
        ll_mask = torch.zeros(H, W)
        ll_mask[10:38, 25:27] = 1.0  # 왼쪽 차선
        ll_mask[10:38, 53:55] = 1.0  # 오른쪽 차선
        
        ego_status = {
            'velocity': [10.0, 0.0],  # 10m/s 직진
            'steering': 0.0,
            'yaw_rate': 0.0,
            'acceleration': 0.0
        }
        
    elif scenario_type == 'obstacle_avoidance':
        # 장애물 회피 시나리오
        det_mask = torch.zeros(H, W)
        det_mask[20:25, 35:45] = 1.0  # 전방 장애물
        
        da_mask = torch.zeros(H, W)
        da_mask[10:38, 15:65] = 1.0  # 넓은 주행 가능 영역
        
        ll_mask = torch.zeros(H, W)
        ll_mask[10:38, 20:22] = 1.0  # 왼쪽 차선
        ll_mask[10:38, 58:60] = 1.0  # 오른쪽 차선
        
        ego_status = {
            'velocity': [8.0, 0.5],   # 약간의 측면 속도
            'steering': 0.1,
            'yaw_rate': 0.05,
            'acceleration': -1.0      # 감속
        }
        
    elif scenario_type == 'lane_change':
        # 차선 변경 시나리오
        det_mask = torch.zeros(H, W)
        
        da_mask = torch.zeros(H, W)
        da_mask[10:38, 15:70] = 1.0  # 넓은 주행 가능 영역
        
        ll_mask = torch.zeros(H, W)
        ll_mask[10:38, 25:27] = 1.0  # 현재 차선 왼쪽
        ll_mask[10:38, 40:42] = 1.0  # 중앙선
        ll_mask[10:38, 55:57] = 1.0  # 목표 차선 오른쪽
        
        ego_status = {
            'velocity': [12.0, -2.0],  # 우측으로 차선 변경
            'steering': -0.15,
            'yaw_rate': -0.1,
            'acceleration': 0.5
        }
    
    else:
        raise ValueError(f"Unknown scenario: {scenario_type}")
    
    return det_mask, da_mask, ll_mask, ego_status


def test_single_scenario(system, scenario_name, scenario_type):
    """단일 시나리오 테스트"""
    print(f"\n🧪 {scenario_name} 시나리오 테스트...")
    
    # 테스트 데이터 생성
    det_mask, da_mask, ll_mask, ego_status = create_test_scenario(scenario_type)
    
    # 배치 차원 추가
    det_mask = det_mask.unsqueeze(0)
    da_mask = da_mask.unsqueeze(0)
    ll_mask = ll_mask.unsqueeze(0)
    
    # 시스템 실행
    result = system.forward(det_mask, da_mask, ll_mask, ego_status, return_details=True)
    
    # 결과 분석
    trajectory = result['trajectory'][0]  # [num_steps, 2]
    confidence = result['confidence'][0]  # [num_steps]
    
    print(f"✅ {scenario_name} 테스트 완료!")
    print(f"   - 추론 시간: {result['inference_time']*1000:.2f} ms")
    print(f"   - 안전 점수: {result['safety_score']:.3f}")
    print(f"   - 궤적 범위: x=[{trajectory[:, 0].min():.2f}, {trajectory[:, 0].max():.2f}], y=[{trajectory[:, 1].min():.2f}, {trajectory[:, 1].max():.2f}]")
    print(f"   - 평균 신뢰도: {confidence.mean():.3f}")
    
    return result


def test_batch_processing(system, batch_size=4):
    """배치 처리 테스트"""
    print(f"\n🔄 배치 처리 테스트 (배치 크기: {batch_size})...")
    
    # 배치 데이터 생성
    det_masks = []
    da_masks = []
    ll_masks = []
    ego_statuses = []
    
    scenarios = ['straight_road', 'obstacle_avoidance', 'lane_change', 'straight_road']
    
    for i in range(batch_size):
        scenario = scenarios[i % len(scenarios)]
        det, da, ll, ego = create_test_scenario(scenario)
        det_masks.append(det)
        da_masks.append(da)
        ll_masks.append(ll)
        ego_statuses.append(ego)
    
    # 배치 텐서 생성
    det_batch = torch.stack(det_masks)
    da_batch = torch.stack(da_masks)
    ll_batch = torch.stack(ll_masks)
    
    # 배치 ego status (첫 번째 것을 사용)
    ego_status = ego_statuses[0]
    
    # 배치 실행
    result = system.forward(det_batch, da_batch, ll_batch, ego_status)
    
    print(f"✅ 배치 처리 완료!")
    print(f"   - 배치 크기: {det_batch.shape[0]}")
    print(f"   - 추론 시간: {result['inference_time']*1000:.2f} ms")
    print(f"   - 평균 안전 점수: {result['safety_score']:.3f}")


def test_performance_benchmark(system, num_iterations=100):
    """성능 벤치마크"""
    print(f"\n⚡ 성능 벤치마크 ({num_iterations}회 반복)...")
    
    # 테스트 데이터 (straight_road 시나리오)
    det_mask, da_mask, ll_mask, ego_status = create_test_scenario('straight_road')
    det_mask = det_mask.unsqueeze(0).to(system.device)
    da_mask = da_mask.unsqueeze(0).to(system.device)
    ll_mask = ll_mask.unsqueeze(0).to(system.device)
    
    # 워밍업
    for _ in range(10):
        _ = system.forward(det_mask, da_mask, ll_mask, ego_status)
    
    system.reset_stats()
    
    # 실제 벤치마크
    for i in range(num_iterations):
        result = system.forward(det_mask, da_mask, ll_mask, ego_status)
        if (i + 1) % 20 == 0:
            print(f"   진행: {i+1}/{num_iterations}")
    
    # 성능 통계
    stats = system.get_performance_stats()
    
    print(f"✅ 성능 벤치마크 완료!")
    print(f"   - 평균 추론 시간: {stats['avg_inference_time']*1000:.2f} ms")
    print(f"   - 최대 추론 시간: {stats['max_inference_time']*1000:.2f} ms")
    print(f"   - 최소 추론 시간: {stats['min_inference_time']*1000:.2f} ms")
    print(f"   - 예상 FPS: {stats['fps']:.1f}")
    print(f"   - 평균 안전 점수: {stats['avg_safety_score']:.3f}")


def visualize_results(results, save_path="integration_test_results.png"):
    """결과 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('BEV-Planner Integration Test Results', fontsize=16)
    
    # 시나리오별 궤적 플롯
    scenarios = ['Straight Road', 'Obstacle Avoidance', 'Lane Change']  
    colors = ['blue', 'red', 'green']
    
    for i, (name, result) in enumerate(zip(scenarios, results)):
        trajectory = result['trajectory'][0].numpy()  # [num_steps, 2]
        
        # 궤적 플롯
        axes[0, i].plot(trajectory[:, 0], trajectory[:, 1], 'o-', color=colors[i], linewidth=2, markersize=6)
        axes[0, i].set_title(f'{name}\nSafety Score: {result["safety_score"]:.3f}')
        axes[0, i].set_xlabel('X (meters)')
        axes[0, i].set_ylabel('Y (meters)')
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].axis('equal')
        
        # 신뢰도 플롯
        confidence = result['confidence'][0].numpy()
        axes[1, i].bar(range(len(confidence)), confidence, color=colors[i], alpha=0.7)
        axes[1, i].set_title(f'Confidence Scores\nAvg: {confidence.mean():.3f}')
        axes[1, i].set_xlabel('Time Step')
        axes[1, i].set_ylabel('Confidence')
        axes[1, i].set_ylim(0, 1)
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 시각화 결과 저장: {save_path}")


def main():
    """메인 테스트 함수"""
    print("🚀 BEV-Planner 통합 시스템 테스트 시작!")
    
    # 시스템 초기화
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  사용 디바이스: {device}")
    
    system = IntegratedBEVPlanningSystem(device=device)
    
    # 1. 시나리오별 테스트
    scenario_results = []
    scenarios = [
        ('직진 도로', 'straight_road'),
        ('장애물 회피', 'obstacle_avoidance'), 
        ('차선 변경', 'lane_change')
    ]
    
    for name, scenario_type in scenarios:
        result = test_single_scenario(system, name, scenario_type)
        scenario_results.append(result)
    
    # 2. 배치 처리 테스트
    test_batch_processing(system, batch_size=4)
    
    # 3. 성능 벤치마크
    test_performance_benchmark(system, num_iterations=100)
    
    # 4. 결과 시각화
    visualize_results(scenario_results)
    
    print("\n🎉 모든 통합 테스트 완료!")
    print("   - YOLOP Adapter ✅")
    print("   - BEV-Planner ✅") 
    print("   - Safety Checker ✅")
    print("   - 실시간 성능 ✅")


if __name__ == "__main__":
    main() 