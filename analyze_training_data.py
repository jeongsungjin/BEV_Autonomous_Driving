#!/usr/bin/env python3
"""
BEV-Planner 학습 데이터 분석 스크립트
"""

import os
import pickle
import glob
from datetime import datetime
import statistics

def analyze_training_data(data_dir):
    """학습 데이터 분석"""
    pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))
    pkl_files.sort()
    
    print("="*60)
    print("📊 BEV-Planner 학습 데이터 분석 보고서")
    print("="*60)
    
    total_samples = 0
    total_size = 0
    velocities = []
    angular_velocities = []
    
    print(f"\n📁 데이터 파일 수: {len(pkl_files)}")
    print(f"📂 총 데이터 용량: {sum(os.path.getsize(f) for f in pkl_files) / (1024*1024):.1f} MB")
    print("\n📋 파일별 상세 정보:")
    
    for i, pkl_file in enumerate(pkl_files):
        file_size = os.path.getsize(pkl_file) / (1024*1024)  # MB
        filename = os.path.basename(pkl_file)
        
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                
            samples_count = len(data)
            total_samples += samples_count
            total_size += file_size
            
            # 데이터 품질 분석
            file_velocities = [sample['ego_velocity'] for sample in data if 'ego_velocity' in sample]
            file_angular_velocities = [sample['ego_angular_velocity'] for sample in data if 'ego_angular_velocity' in sample]
            
            velocities.extend(file_velocities)
            angular_velocities.extend(file_angular_velocities)
            
            # 각속도 0인 비율 계산
            zero_angular_count = sum(1 for av in file_angular_velocities if abs(av) < 1e-6)
            zero_angular_ratio = zero_angular_count / len(file_angular_velocities) * 100 if file_angular_velocities else 0
            
            print(f"  {i+1:2d}. {filename}")
            print(f"      📊 샘플 수: {samples_count:,}")
            print(f"      💾 파일 크기: {file_size:.1f} MB")
            print(f"      🏃 평균 속도: {statistics.mean(file_velocities):.2f} m/s")
            print(f"      🔄 각속도 0 비율: {zero_angular_ratio:.1f}%")
            print()
            
        except Exception as e:
            print(f"  ❌ {filename}: 읽기 실패 - {e}")
    
    # 전체 통계
    print("🎯 전체 데이터 통계:")
    print(f"  📊 총 샘플 수: {total_samples:,}")
    print(f"  💾 총 용량: {total_size:.1f} MB")
    print(f"  📈 샘플당 평균 크기: {total_size*1024/total_samples:.1f} KB")
    
    if velocities and angular_velocities:
        print(f"\n🚗 주행 데이터 품질:")
        print(f"  🏃 속도 범위: {min(velocities):.2f} ~ {max(velocities):.2f} m/s")
        print(f"  🏃 평균 속도: {statistics.mean(velocities):.2f} m/s")
        print(f"  🔄 각속도 범위: {min(angular_velocities):.3f} ~ {max(angular_velocities):.3f} rad/s")
        print(f"  🔄 평균 각속도: {statistics.mean(angular_velocities):.3f} rad/s")
        
        # 각속도 0인 비율
        zero_angular_count = sum(1 for av in angular_velocities if abs(av) < 1e-6)
        zero_angular_ratio = zero_angular_count / len(angular_velocities) * 100
        print(f"  ⚠️  각속도 0 비율: {zero_angular_ratio:.1f}%")
        
        # 각속도 분포 분석
        positive_angular = sum(1 for av in angular_velocities if av > 0.05)
        negative_angular = sum(1 for av in angular_velocities if av < -0.05)
        print(f"  ↰ 우회전 샘플: {positive_angular:,} ({positive_angular/len(angular_velocities)*100:.1f}%)")
        print(f"  ↱ 좌회전 샘플: {negative_angular:,} ({negative_angular/len(angular_velocities)*100:.1f}%)")
    
    # 학습 권장사항
    print(f"\n🎓 학습 권장사항:")
    
    if total_samples < 500:
        print("  ❌ 데이터 부족: 최소 500-1000 샘플 필요")
        status = "부족"
    elif total_samples < 1000:
        print("  ⚠️  데이터 최소: 1000-2000 샘플 권장")
        status = "최소"
    elif total_samples < 2000:
        print("  ✅ 데이터 적당: 학습 가능")
        status = "적당"
    else:
        print("  🎉 데이터 충분: 좋은 성능 기대")
        status = "충분"
    
    if angular_velocities:
        if zero_angular_ratio > 80:
            print("  ❌ 각속도 문제: 대부분이 직진 데이터 (커브 데이터 필요)")
        elif zero_angular_ratio > 60:
            print("  ⚠️  각속도 주의: 커브 데이터 비율 증가 필요")
        else:
            print("  ✅ 각속도 양호: 다양한 주행 패턴 포함")
    
    print("\n💡 데이터 수집 팁:")
    print("  - 직진, 좌회전, 우회전, 차선변경을 골고루")
    print("  - 다양한 속도로 주행 (1-15 m/s)")
    print("  - 최소 1000 샘플, 권장 2000+ 샘플")
    
    return {
        'total_samples': total_samples,
        'total_size_mb': total_size,
        'status': status,
        'zero_angular_ratio': zero_angular_ratio if angular_velocities else 100,
        'avg_velocity': statistics.mean(velocities) if velocities else 0,
        'avg_angular_velocity': statistics.mean(angular_velocities) if angular_velocities else 0
    }

if __name__ == '__main__':
    data_dir = "training_data"
    if not os.path.exists(data_dir):
        print(f"❌ 데이터 디렉터리가 없습니다: {data_dir}")
        exit(1)
    
    stats = analyze_training_data(data_dir)
    print("="*60) 