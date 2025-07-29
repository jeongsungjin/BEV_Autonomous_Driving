#!/usr/bin/env python3
"""
BEV-Planner 모델 학습 스크립트
CARLA에서 수집한 데이터로 BEV-Planner를 학습시킵니다.
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# 프로젝트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.simplified_planner import SimplifiedBEVPlanner
from models.safety_checker import PlanningLoss
from utils.dataset import create_dataloaders, BEVNormalize, EgoNormalize
from adapters.yolop_adapter import YOLOPToBEVAdapter

class BEVPlannerTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 체크포인트 디렉터리 설정
        self.checkpoint_dir = os.path.join(config['save_dir'], 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # TensorBoard 설정
        self.writer = SummaryWriter(
            log_dir=os.path.join(config['save_dir'], 'tensorboard')
        )
        
        self.logger.info(f"🖥️  사용 디바이스: {self.device}")
        self.logger.info(f"📁 체크포인트 저장 경로: {self.checkpoint_dir}")
        
        # 모델, 데이터, 옵티마이저 초기화
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        self.setup_loss()
        
        # 학습 상태
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0
        
    def setup_model(self):
        """모델 초기화"""
        # YOLOP adapter
        self.adapter = YOLOPToBEVAdapter(
            input_height=48,
            input_width=80,
            embed_dim=self.config['bev_embed_dim']
        ).to(self.device)
        
        # BEV Planner
        self.model = SimplifiedBEVPlanner(
            bev_embed_dim=self.config['bev_embed_dim'],
            ego_embed_dim=self.config['ego_embed_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_future_steps=self.config['num_future_steps'],
            num_transformer_layers=self.config['num_transformer_layers'],
            num_attention_heads=self.config['num_attention_heads'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        self.logger.info(f"✅ 모델 초기화 완료")
        self.logger.info(f"📊 모델 파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def setup_data(self):
        """데이터 로더 설정"""
        # 데이터 변환
        transform = None
        if self.config.get('normalize', True):
            # TODO: 실제 데이터 통계 기반으로 정규화 값 계산
            transform = lambda bev, ego, traj: (
                BEVNormalize()(bev, ego, traj)[0:1] + 
                EgoNormalize()(bev, ego, traj)[1:3]
            )
        
        self.train_loader, self.val_loader = create_dataloaders(
            data_dir=self.config['data_dir'],
            batch_size=self.config['batch_size'],
            val_split=self.config['val_split'],
            num_workers=self.config['num_workers'],
            max_samples=self.config.get('max_samples')
        )
        
        self.logger.info(f"📚 학습 배치 수: {len(self.train_loader)}")
        self.logger.info(f"📖 검증 배치 수: {len(self.val_loader)}")
        
    def setup_optimizer(self):
        """옵티마이저 설정"""
        # 모든 모델 파라미터 합치기
        all_parameters = list(self.adapter.parameters()) + list(self.model.parameters())
        
        self.optimizer = optim.AdamW(
            all_parameters,
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # 학습률 스케줄러
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=self.config['scheduler_patience'],
            verbose=True
        )
        
        self.logger.info(f"⚙️  옵티마이저: AdamW (lr={self.config['learning_rate']})")
        
    def setup_loss(self):
        """손실 함수 설정"""
        self.planning_loss = PlanningLoss(
            trajectory_weight=self.config['trajectory_weight'],
            collision_weight=self.config['collision_weight'],
            lane_keeping_weight=self.config['lane_keeping_weight'],
            smoothness_weight=self.config['smoothness_weight'],
            confidence_weight=self.config['confidence_weight']
        )
        
        self.mse_loss = nn.MSELoss()
        
        self.logger.info("✅ 손실 함수 설정 완료")
        
    def train_epoch(self):
        """한 에포크 학습"""
        self.adapter.train()
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, (bev_features, ego_features, target_trajectory) in enumerate(progress_bar):
            # 데이터를 GPU로 이동
            bev_features = bev_features.to(self.device)  # (B, 3, 48, 80)
            ego_features = ego_features.to(self.device)  # (B, 2)
            target_trajectory = target_trajectory.to(self.device)  # (B, 6, 2)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Ego status를 딕셔너리 형태로 변환
            ego_status_dict = {
                'velocity': [ego_features[0, 0].item(), 0.0],  # [vx, vy]
                'steering': 0.0,
                'yaw_rate': ego_features[0, 1].item(),  # angular_velocity
                'acceleration': 0.0
            }
            
            # YOLOP features를 BEV features로 변환
            adapted_bev_features = self.adapter(
                detection_mask=bev_features[:, 0],  # (B, 48, 80)
                drivable_area_mask=bev_features[:, 1],   # (B, 48, 80)
                lane_line_mask=bev_features[:, 2],   # (B, 48, 80)
                ego_status=ego_status_dict
            )
            
            # BEV-Planner 추론
            outputs = self.model(
                bev_features=adapted_bev_features['bev_features'],
                ego_features=ego_features
            )
            
            predicted_trajectory = outputs['trajectory']  # (B, 6, 2)
            predicted_confidence = outputs['confidence']  # (B, 6)
            
            # 손실 계산
            # 1. MSE Loss (기본 궤적 매칭)
            mse_loss = self.mse_loss(predicted_trajectory, target_trajectory)
            
            # 2. Planning Loss (물리적 제약, 충돌 등)  
            planning_loss_value = self.planning_loss(
                predicted_trajectory=predicted_trajectory,
                predicted_confidence=predicted_confidence,
                target_trajectory=target_trajectory,
                detection_mask=bev_features[:, 0] > 0.5,  # (B, 48, 80)
                drivable_area_mask=bev_features[:, 1] > 0.5,  # (B, 48, 80)
                lane_line_mask=bev_features[:, 2] > 0.5   # (B, 48, 80)
            )
            
            # 총 손실
            total_loss_batch = mse_loss + planning_loss_value['total_loss']
            
            # Backward pass
            total_loss_batch.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(
                list(self.adapter.parameters()) + list(self.model.parameters()),
                max_norm=self.config['grad_clip']
            )
            
            self.optimizer.step()
            
            # 통계 업데이트
            total_loss += total_loss_batch.item()
            
            # 진행률 업데이트
            progress_bar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
                'MSE': f'{mse_loss.item():.4f}',
                'Plan': f'{planning_loss_value["total_loss"].item():.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # TensorBoard 로깅
            if self.global_step % self.config['log_interval'] == 0:
                self.writer.add_scalar('Train/Total_Loss', total_loss_batch.item(), self.global_step)
                self.writer.add_scalar('Train/MSE_Loss', mse_loss.item(), self.global_step)
                self.writer.add_scalar('Train/Planning_Loss', planning_loss_value['total_loss'].item(), self.global_step)
                self.writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], self.global_step)
                
                # 세부 planning loss들도 로깅
                for loss_name, loss_value in planning_loss_value.items():
                    if loss_name != 'total_loss' and loss_value is not None:
                        self.writer.add_scalar(f'Train/Planning_{loss_name}', loss_value.item(), self.global_step)
                
            self.global_step += 1
            
        avg_loss = total_loss / num_batches
        self.logger.info(f"📊 Epoch {self.current_epoch+1} 학습 완료 - 평균 손실: {avg_loss:.4f}")
        
        return avg_loss
        
    def validate(self):
        """검증"""
        self.adapter.eval()
        self.model.eval()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for bev_features, ego_features, target_trajectory in self.val_loader:
                # 데이터를 GPU로 이동
                bev_features = bev_features.to(self.device)
                ego_features = ego_features.to(self.device)
                target_trajectory = target_trajectory.to(self.device)
                
                # Ego status를 딕셔너리 형태로 변환
                ego_status_dict = {
                    'velocity': [ego_features[0, 0].item(), 0.0],  # [vx, vy]
                    'steering': 0.0,
                    'yaw_rate': ego_features[0, 1].item(),  # angular_velocity
                    'acceleration': 0.0
                }
                
                # Forward pass
                adapted_bev_features = self.adapter(
                    detection_mask=bev_features[:, 0],
                    drivable_area_mask=bev_features[:, 1],
                    lane_line_mask=bev_features[:, 2],
                    ego_status=ego_status_dict
                )
                
                outputs = self.model(
                    bev_features=adapted_bev_features['bev_features'],
                    ego_features=ego_features
                )
                
                predicted_trajectory = outputs['trajectory']
                predicted_confidence = outputs['confidence']
                
                # 손실 계산
                mse_loss = self.mse_loss(predicted_trajectory, target_trajectory)
                
                planning_loss_value = self.planning_loss(
                    predicted_trajectory=predicted_trajectory,
                    predicted_confidence=predicted_confidence,
                    target_trajectory=target_trajectory,
                    detection_mask=bev_features[:, 0] > 0.5,
                    drivable_area_mask=bev_features[:, 1] > 0.5,
                    lane_line_mask=bev_features[:, 2] > 0.5
                )
                
                total_loss_batch = mse_loss + planning_loss_value['total_loss']
                total_loss += total_loss_batch.item()
                
        avg_val_loss = total_loss / num_batches
        self.logger.info(f"🔍 검증 완료 - 평균 손실: {avg_val_loss:.4f}")
        
        # TensorBoard 로깅
        self.writer.add_scalar('Val/Loss', avg_val_loss, self.current_epoch)
        
        return avg_val_loss
        
    def save_checkpoint(self, is_best=False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': self.current_epoch,
            'adapter_state_dict': self.adapter.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
            'config': self.config
        }
        
        # 최신 체크포인트 저장
        checkpoint_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 최고 성능 체크포인트 저장
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"🏆 최고 성능 체크포인트 저장: {best_path}")
            
        # 에포크별 체크포인트 저장 (선택적)
        if self.current_epoch % self.config['save_interval'] == 0:
            epoch_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{self.current_epoch}.pth')
            torch.save(checkpoint, epoch_path)
            
    def load_checkpoint(self, checkpoint_path):
        """체크포인트 로드"""
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"⚠️  체크포인트 파일이 없습니다: {checkpoint_path}")
            return False
            
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.adapter.load_state_dict(checkpoint['adapter_state_dict'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.current_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.global_step = checkpoint['global_step']
            
            self.logger.info(f"✅ 체크포인트 로드 완료: epoch {self.current_epoch}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 체크포인트 로드 실패: {e}")
            return False
            
    def train(self):
        """메인 학습 루프"""
        self.logger.info("🚀 BEV-Planner 학습 시작!")
        
        # 체크포인트에서 재시작 (옵션)
        if self.config.get('resume_from_checkpoint'):
            self.load_checkpoint(self.config['resume_from_checkpoint'])
            
        try:
            for epoch in range(self.current_epoch, self.config['num_epochs']):
                self.current_epoch = epoch
                
                # 학습
                train_loss = self.train_epoch()
                
                # 검증
                val_loss = self.validate()
                
                # 학습률 스케줄링
                self.scheduler.step(val_loss)
                
                # 최고 성능 확인
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    
                # 체크포인트 저장
                self.save_checkpoint(is_best=is_best)
                
                self.logger.info(f"🎯 Epoch {epoch+1} 완료 - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
                
        except KeyboardInterrupt:
            self.logger.info("⏹️  학습이 중단되었습니다")
        except Exception as e:
            import traceback
            self.logger.error(f"❌ 학습 중 오류 발생: {e}")
            self.logger.error(f"상세 오류:\n{traceback.format_exc()}")
        finally:
            self.writer.close()
            
        self.logger.info("✅ 학습 완료!")
        self.logger.info(f"🏆 최고 검증 손실: {self.best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='BEV-Planner 학습')
    parser.add_argument('--config', type=str, default='config/train_config.json',
                       help='설정 파일 경로')
    parser.add_argument('--data_dir', type=str, 
                       default=os.path.expanduser('~/capstone_2025/training_data'),
                       help='학습 데이터 디렉터리')
    parser.add_argument('--save_dir', type=str,
                       default=os.path.expanduser('~/capstone_2025/training_results'),
                       help='결과 저장 디렉터리')
    
    args = parser.parse_args()
    
    # 기본 설정
    default_config = {
        'data_dir': args.data_dir,
        'save_dir': args.save_dir,
        'batch_size': 8,   # GPU 메모리 절약을 위해 줄임
        'num_epochs': 50,  # 초기 학습은 50 에포크로 시작
        'learning_rate': 5e-5,  # 조금 더 보수적인 학습률
        'weight_decay': 1e-4,   # 정규화 강화
        'val_split': 0.2,
        'num_workers': 2,  # 안정성을 위해 줄임
        
        # 모델 설정
        'bev_embed_dim': 256,
        'ego_embed_dim': 2,  # ego_features는 [velocity, angular_velocity] 2차원
        'hidden_dim': 512,
        'num_future_steps': 6,
        'num_transformer_layers': 4,
        'num_attention_heads': 8,
        'dropout': 0.1,
        
        # 손실 가중치 (각속도 개선을 위해 조정)
        'trajectory_weight': 2.0,     # 궤적 정확도 강화
        'collision_weight': 8.0,      # 안전성 유지하되 너무 과하지 않게
        'lane_keeping_weight': 3.0,   # 차선 유지 적당히
        'smoothness_weight': 5.0,     # 부드러운 경로 생성 강화
        'confidence_weight': 0.5,     # 신뢰도 가중치 증가
        
        # 기타
        'grad_clip': 0.5,       # 그래디언트 클리핑 강화
        'log_interval': 20,     # 더 자주 로깅
        'save_interval': 5,     # 더 자주 체크포인트 저장
        'scheduler_patience': 3, # 학습률 감소 조건 엄격화
        'max_samples': None,
        'normalize': True,
        'early_stopping_patience': 10  # 조기 종료 조건 추가
    }
    
    # 설정 파일 로드 (존재할 경우)
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            default_config.update(file_config)
    
    # 디렉터리 생성
    os.makedirs(default_config['save_dir'], exist_ok=True)
    
    # 설정 저장
    config_save_path = os.path.join(default_config['save_dir'], 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    # 학습 시작
    trainer = BEVPlannerTrainer(default_config)
    trainer.train()


if __name__ == '__main__':
    main() 