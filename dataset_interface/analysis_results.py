import os
import sys
import torch
import numpy as np
from collections import namedtuple


env_path = os.path.dirname(os.path.abspath(__file__))
if env_path not in sys.path:
    sys.path.append(env_path)

from evaluation import get_dataset
from analysis.extract_results import extract_results
from analysis.plot_results import get_auc_curve, get_prec_curve
from evaluation.environment import env_settings
import argparse

SimpleTracker = namedtuple('SimpleTracker', ['name', 'parameter_name', 'run_id', 'results_dir', 'display_name'])

def evaluate_direct(results_path, dataset_name):
    """直接评估预测结果
    
    Args:
        results_path: 预测结果txt文件所在的目录
        dataset_name: 数据集名称
    """

    dataset = get_dataset(dataset_name)
    # print(f"成功加载数据集: {dataset_name}, 共有 {len(dataset)} 个序列")
    print(f"sucessfully load dataset: {dataset_name}, total {len(dataset)} sequences")

    
    # 3. 创建一个简单的tracker对象
    # 注意：extract_results假设results_dir下有dataset_name的子目录
    # 但你的路径可能已经包含了这个结构，所以我们取最后一级目录作为parameter_name
    parameter_name = os.path.basename(results_path) if results_path.endswith('/') else os.path.basename(results_path)
    parent_dir = os.path.dirname(results_path)
    
    tracker = SimpleTracker(
        name='custom_tracker',
        parameter_name=parameter_name,
        run_id=None,
        results_dir=parent_dir,  # 使用父目录作为results_dir
        display_name='CustomTracker'
    )
    
    report_name = f"{dataset_name}_{parameter_name}_eval"
    eval_data = extract_results(
        trackers=[tracker], 
        dataset=dataset, 
        report_name=report_name,
        skip_missing_seq=True,  # 跳过缺失的序列
        plot_bin_gap=0.05,
        exclude_invalid_frames=False
    )
    
    print("\n evaluate done!")
    

    if eval_data:
        valid_sequence = torch.tensor(eval_data['valid_sequence'])
        if not valid_sequence.any():
            print("warning: no valid sequences found.")
            return
        
        # AUC (Success) 计算
        ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])
        auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
        
        # Precision 计算
        ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
        
        # Normalized Precision 计算
        ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])
        norm_prec_curve, norm_prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
        
        print(f"dataset name: {dataset_name}")
        print(f"evaled sequence num: {valid_sequence.sum().item()} / {len(dataset)}")
        print(f"Success (AUC): {auc.item():.4f}")
        print(f"Precision: {prec_score.item():.4f}")
        print(f"Normalized Precision: {norm_prec_score.item():.4f}")
        
        # 输出保存位置
        settings = env_settings()
        result_path = os.path.join(settings.result_plot_path, report_name)
        print(f"\n results saved in: {result_path}/eval_data.pkl")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate tracking results directly from prediction files.')
    parser.add_argument('results_path', type=str, help='Path to the directory containing prediction txt files.')
    parser.add_argument('dataset_name', type=str, help='Name of the dataset to evaluate on.')
    
    args = parser.parse_args()
    
    my_results_path = args.results_path
    my_dataset_name = args.dataset_name
    
    evaluate_direct(my_results_path, my_dataset_name)

