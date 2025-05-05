#验证SDK token
from modelscope.hub.api import HubApi
api = HubApi()
api.login('470dc5bd-b4d9-4ed2-997b-a3684903260e')

#数据集下载
from modelscope.msdatasets import MsDataset
# Specify the download path using the target_dir parameter
ds =  MsDataset.load('FineTuneWata/tracking_data', target_dir='/data1/lihaobo/tracking/data1')

# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# from huggingface_hub import snapshot_download
# snapshot_download(repo_id='l-lt/LaSOT', repo_type='dataset', local_dir='./data1')
