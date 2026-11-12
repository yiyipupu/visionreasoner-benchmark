import os
from pathlib import Path
root=Path(os.environ.get('VISIONREASONER_ROOT','/data/yukun/VisionReasoner'))
os.environ.setdefault('HF_HOME',str(root/'cache/huggingface'))
from huggingface_hub import snapshot_download
for repo in ('Ricky06662/VisionReasoner-7B','Ricky06662/TaskRouter-1.5B','facebook/sam2-hiera-large'):
    snapshot_download(repo,local_dir=str(root/'weights'/repo.split('/')[-1]),max_workers=1)
