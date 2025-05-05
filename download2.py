from modelscope.msdatasets import MsDataset
import os
import zipfile

def download_folder_dataset(repo_id, target_dir):
    # 获取数据集元数据
    dataset = MsDataset.load(repo_id, download_mode='force_redownload')
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 递归处理文件结构
    for split in dataset._data_files:
        for file_info in split['files']:
            remote_path = file_info['path']
            local_path = os.path.join(target_dir, remote_path)
            
            # 创建本地目录结构
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # 下载文件
            with open(local_path, 'wb') as f:
                f.write(dataset.get_file(remote_path).getbuffer())
                
            # 自动解压嵌套ZIP
            if local_path.endswith('.zip'):
                with zipfile.ZipFile(local_path, 'r') as z:
                    z.extractall(os.path.dirname(local_path))
                os.remove(local_path)  # 可选删除原ZIP

# 使用示例
download_folder_dataset('FineTuneWata/tracking_data', './complex_dataset')
