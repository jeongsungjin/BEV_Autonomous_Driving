#!/usr/bin/env python3
"""
YOLOP 어댑터 테스트 스크립트
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from yolop_adapter import YOLOPToBEVAdapter, BEVFeatureProcessor


def create_mock_yolop_output(height=48, width=80):
    """테스트용 가짜 YOLOP 출력 생성"""
    
    # Detection mask - 중앙에 몇 개의 객체
    det_mask = np.zeros((height, width), dtype=np.float32)
    det_mask[20:25, 30:35] = 1.0  # 객체 1
    det_mask[15:20, 50:55] = 1.0  # 객체 2
    
    # Drivable area mask - 중앙 도로 영역
    da_mask = np.zeros((height, width), dtype=np.float32)
    da_mask[10:38, 15:65] = 1.0  # 주행 가능 영역
    
    # Lane line mask - 차선
    ll_mask = np.zeros((height, width), dtype=np.float32)
    ll_mask[10:38, 20:22] = 1.0  # 왼쪽 차선
    ll_mask[10:38, 58:60] = 1.0  # 오른쪽 차선
    ll_mask[10:38, 38:40] = 1.0  # 중앙선
    
    return torch.from_numpy(det_mask), torch.from_numpy(da_mask), torch.from_numpy(ll_mask)


def test_adapter():
    """어댑터 기본 기능 테스트"""
    print("🧪 YOLOP 어댑터 테스트 시작...")
    
    # 1. 어댑터 초기화
    adapter = YOLOPToBEVAdapter(
        input_height=48,
        input_width=80,
        embed_dim=256,
        use_positional_encoding=True
    )
    
    print(f"✅ 어댑터 초기화 완료 - 임베딩 차원: {adapter.embed_dim}")
    
    # 2. 가짜 YOLOP 출력 생성
    det_mask, da_mask, ll_mask = create_mock_yolop_output()
    print(f"✅ 테스트 데이터 생성 완료 - 크기: {det_mask.shape}")
    
    # 3. Ego 상태 생성
    ego_status = BEVFeatureProcessor.create_mock_ego_status(
        velocity_x=10.0,  # 10 m/s 전진
        velocity_y=0.5,   # 0.5 m/s 측면
        steering=0.1      # 0.1 rad 조향
    )
    print(f"✅ Ego 상태 생성 완료: {ego_status}")
    
    # 4. 어댑터 실행
    with torch.no_grad():
        result = adapter(det_mask, da_mask, ll_mask, ego_status)
    
    print(f"✅ 어댑터 실행 완료!")
    print(f"   - BEV features 크기: {result['bev_features'].shape}")
    print(f"   - Ego features 크기: {result['ego_features'].shape}")
    
    # 5. 결과 분석
    bev_feat = result['bev_features']  # [1, H*W, C]
    ego_feat = result['ego_features']  # [1, C]
    
    print(f"\n📊 결과 분석:")
    print(f"   - BEV 특징 범위: [{bev_feat.min().item():.3f}, {bev_feat.max().item():.3f}]")
    print(f"   - BEV 특징 평균: {bev_feat.mean().item():.3f}")
    print(f"   - Ego 특징 범위: [{ego_feat.min().item():.3f}, {ego_feat.max().item():.3f}]")
    print(f"   - Ego 특징 평균: {ego_feat.mean().item():.3f}")
    
    return result, (det_mask, da_mask, ll_mask)


def visualize_masks(det_mask, da_mask, ll_mask):
    """마스크 시각화"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(det_mask.numpy(), cmap='Reds')
    axes[0].set_title('Detection Mask')
    axes[0].axis('off')
    
    axes[1].imshow(da_mask.numpy(), cmap='Greens')
    axes[1].set_title('Drivable Area Mask')
    axes[1].axis('off')
    
    axes[2].imshow(ll_mask.numpy(), cmap='Blues')
    axes[2].set_title('Lane Line Mask')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_masks.png', dpi=150, bbox_inches='tight')
    print("📸 마스크 시각화 저장: test_masks.png")


def test_batch_processing():
    """배치 처리 테스트"""
    print("\n🔄 배치 처리 테스트...")
    
    adapter = YOLOPToBEVAdapter()
    
    # 배치 데이터 생성 (배치 크기 = 3)
    batch_size = 3
    det_masks = []
    da_masks = []
    ll_masks = []
    
    for i in range(batch_size):
        det, da, ll = create_mock_yolop_output()
        det_masks.append(det)
        da_masks.append(da)
        ll_masks.append(ll)
    
    det_batch = torch.stack(det_masks)  # [3, H, W]
    da_batch = torch.stack(da_masks)   # [3, H, W]
    ll_batch = torch.stack(ll_masks)   # [3, H, W]
    
    ego_status = BEVFeatureProcessor.create_mock_ego_status()
    
    with torch.no_grad():
        result = adapter(det_batch, da_batch, ll_batch, ego_status)
    
    print(f"✅ 배치 처리 완료!")
    print(f"   - 입력 배치 크기: {det_batch.shape}")
    print(f"   - BEV features 크기: {result['bev_features'].shape}")
    print(f"   - Ego features 크기: {result['ego_features'].shape}")


def test_performance():
    """성능 테스트"""
    print("\n⚡ 성능 테스트...")
    
    adapter = YOLOPToBEVAdapter()
    det_mask, da_mask, ll_mask = create_mock_yolop_output()
    ego_status = BEVFeatureProcessor.create_mock_ego_status()
    
    # GPU 사용 가능하면 GPU로 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adapter = adapter.to(device)
    det_mask = det_mask.to(device)
    da_mask = da_mask.to(device)
    ll_mask = ll_mask.to(device)
    
    print(f"📱 사용 디바이스: {device}")
    
    # 성능 측정
    import time
    
    # 워밍업
    with torch.no_grad():
        for _ in range(10):
            _ = adapter(det_mask, da_mask, ll_mask, ego_status)
    
    # 실제 측정
    start_time = time.time()
    num_iterations = 100
    
    with torch.no_grad():
        for _ in range(num_iterations):
            result = adapter(det_mask, da_mask, ll_mask, ego_status)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations
    fps = 1.0 / avg_time
    
    print(f"✅ 성능 측정 완료!")
    print(f"   - 평균 처리 시간: {avg_time*1000:.2f} ms")
    print(f"   - 예상 FPS: {fps:.1f}")


if __name__ == "__main__":
    try:
        # 기본 기능 테스트
        result, masks = test_adapter()
        
        # 시각화
        visualize_masks(*masks)
        
        # 배치 처리 테스트
        test_batch_processing()
        
        # 성능 테스트
        test_performance()
        
        print("\n🎉 모든 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc() 