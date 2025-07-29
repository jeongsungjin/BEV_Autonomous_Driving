#!/usr/bin/env python3
"""
BEV-Planner 학습용 PyTorch Dataset
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from typing import List, Dict, Tuple
import logging

class BEVPlannerDataset(Dataset):
    """BEV-Planner 학습용 데이터셋"""
    
    def __init__(self, 
                 data_dir: str, 
                 transform=None,
                 max_samples: int = None,
                 val_split: float = 0.2):
        """
        Args:
            data_dir: 데이터가 저장된 디렉터리
            transform: 데이터 변환 함수
            max_samples: 최대 샘플 수 (메모리 제한용)
            val_split: 검증 데이터 비율
        """
        self.data_dir = data_dir
        self.transform = transform
        self.max_samples = max_samples
        
        # 데이터 로드
        self.samples = self.load_all_data()
        
        if len(self.samples) == 0:
            raise ValueError(f"데이터를 찾을 수 없습니다: {data_dir}")
            
        logging.info(f"📊 총 {len(self.samples)} 샘플 로드됨")
        
        # 데이터 검증
        self.validate_data()
        
    def load_all_data(self) -> List[Dict]:
        """모든 pickle 파일에서 데이터 로드"""
        all_samples = []
        
        # 모든 .pkl 파일 찾기
        pkl_files = glob.glob(os.path.join(self.data_dir, "*.pkl"))
        
        if not pkl_files:
            logging.warning(f"⚠️  {self.data_dir}에 .pkl 파일이 없습니다")
            return []
            
        for pkl_file in pkl_files:
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, list):
                        all_samples.extend(data)
                    else:
                        all_samples.append(data)
                        
                logging.info(f"✅ {os.path.basename(pkl_file)} 로드 완료")
                
            except Exception as e:
                logging.error(f"❌ {pkl_file} 로드 실패: {e}")
                continue
                
        # 최대 샘플 수 제한
        if self.max_samples and len(all_samples) > self.max_samples:
            all_samples = all_samples[:self.max_samples]
            logging.info(f"🔢 샘플 수를 {self.max_samples}개로 제한")
            
        return all_samples
        
    def validate_data(self):
        """데이터 유효성 검사"""
        required_keys = [
            'det_grid', 'da_grid', 'll_grid',
            'ego_velocity', 'ego_angular_velocity', 
            'expert_trajectory'
        ]
        
        valid_samples = []
        
        for sample in self.samples:
            # 필수 키 확인
            if not all(key in sample for key in required_keys):
                continue
                
            # 데이터 타입 및 크기 확인
            try:
                # BEV grids: (48, 80) 크기
                for grid_key in ['det_grid', 'da_grid', 'll_grid']:
                    grid = sample[grid_key]
                    if not isinstance(grid, np.ndarray) or grid.shape != (48, 80):
                        raise ValueError(f"잘못된 그리드 크기: {grid.shape}")
                        
                # Expert trajectory: (6, 2) 크기
                traj = sample['expert_trajectory']
                if not isinstance(traj, np.ndarray) or traj.shape != (6, 2):
                    raise ValueError(f"잘못된 궤적 크기: {traj.shape}")
                    
                # Ego status: 스칼라 값들
                if not isinstance(sample['ego_velocity'], (int, float)):
                    raise ValueError("잘못된 속도 데이터")
                    
                if not isinstance(sample['ego_angular_velocity'], (int, float)):
                    raise ValueError("잘못된 각속도 데이터")
                    
                valid_samples.append(sample)
                
            except Exception as e:
                logging.warning(f"⚠️  샘플 검증 실패: {e}")
                continue
                
        logging.info(f"✅ {len(valid_samples)}/{len(self.samples)} 샘플이 유효함")
        self.samples = valid_samples
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            bev_features: (3, 48, 80) - det, da, ll 그리드들
            ego_features: (2,) - velocity, angular_velocity
            target_trajectory: (6, 2) - 목표 궤적
        """
        sample = self.samples[idx]
        
        # BEV features: (3, 48, 80)
        det_grid = sample['det_grid']
        da_grid = sample['da_grid'] 
        ll_grid = sample['ll_grid']
        
        bev_features = np.stack([det_grid, da_grid, ll_grid], axis=0)
        bev_features = torch.from_numpy(bev_features).float()
        
        # Ego features: (2,)
        ego_features = np.array([
            sample['ego_velocity'],
            sample['ego_angular_velocity']
        ])
        ego_features = torch.from_numpy(ego_features).float()
        
        # Target trajectory: (6, 2)
        target_trajectory = sample['expert_trajectory']
        target_trajectory = torch.from_numpy(target_trajectory).float()
        
        # 데이터 변환 적용
        if self.transform:
            bev_features, ego_features, target_trajectory = self.transform(
                bev_features, ego_features, target_trajectory
            )
            
        return bev_features, ego_features, target_trajectory
        
    def get_statistics(self) -> Dict:
        """데이터셋 통계 정보"""
        if not self.samples:
            return {}
            
        velocities = [s['ego_velocity'] for s in self.samples]
        angular_vels = [s['ego_angular_velocity'] for s in self.samples]
        
        trajectories = np.array([s['expert_trajectory'] for s in self.samples])
        
        stats = {
            'num_samples': len(self.samples),
            'velocity_mean': np.mean(velocities),
            'velocity_std': np.std(velocities),
            'angular_velocity_mean': np.mean(angular_vels),
            'angular_velocity_std': np.std(angular_vels),
            'trajectory_mean': np.mean(trajectories, axis=0),
            'trajectory_std': np.std(trajectories, axis=0),
            'trajectory_max_distance': np.max(np.linalg.norm(trajectories, axis=-1))
        }
        
        return stats


def create_dataloaders(data_dir: str, 
                      batch_size: int = 32,
                      val_split: float = 0.2,
                      num_workers: int = 4,
                      max_samples: int = None) -> Tuple[DataLoader, DataLoader]:
    """학습/검증 DataLoader 생성"""
    
    # 전체 데이터셋 로드
    full_dataset = BEVPlannerDataset(
        data_dir=data_dir,
        max_samples=max_samples
    )
    
    # 학습/검증 분할
    total_samples = len(full_dataset)
    val_samples = int(total_samples * val_split)
    train_samples = total_samples - val_samples
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_samples, val_samples],
        generator=torch.Generator().manual_seed(42)  # 재현 가능한 분할
    )
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    logging.info(f"📚 학습 데이터: {len(train_dataset)} 샘플")
    logging.info(f"📖 검증 데이터: {len(val_dataset)} 샘플")
    
    return train_loader, val_loader


# 데이터 변환 함수들
class BEVNormalize:
    """BEV 그리드 정규화"""
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        
    def __call__(self, bev_features, ego_features, target_trajectory):
        bev_features = (bev_features - self.mean) / self.std
        return bev_features, ego_features, target_trajectory


class EgoNormalize:
    """Ego 상태 정규화"""
    def __init__(self, velocity_scale=10.0, angular_scale=1.0):
        self.velocity_scale = velocity_scale
        self.angular_scale = angular_scale
        
    def __call__(self, bev_features, ego_features, target_trajectory):
        ego_features[0] = ego_features[0] / self.velocity_scale  # velocity
        ego_features[1] = ego_features[1] / self.angular_scale   # angular velocity
        return bev_features, ego_features, target_trajectory


if __name__ == "__main__":
    # 테스트 코드
    import logging
    logging.basicConfig(level=logging.INFO)
    
    data_dir = os.path.expanduser("~/capstone_2025/training_data")
    
    if os.path.exists(data_dir):
        try:
            dataset = BEVPlannerDataset(data_dir)
            print(f"✅ 데이터셋 로드 성공: {len(dataset)} 샘플")
            
            # 첫 번째 샘플 확인
            if len(dataset) > 0:
                bev, ego, traj = dataset[0]
                print(f"BEV features shape: {bev.shape}")
                print(f"Ego features shape: {ego.shape}")
                print(f"Target trajectory shape: {traj.shape}")
                
                # 통계 출력
                stats = dataset.get_statistics()
                for key, value in stats.items():
                    print(f"{key}: {value}")
                    
        except Exception as e:
            print(f"❌ 데이터셋 로드 실패: {e}")
    else:
        print(f"❌ 데이터 디렉터리가 없습니다: {data_dir}")
        print("💡 먼저 data_collector_node.py로 데이터를 수집하세요!") 