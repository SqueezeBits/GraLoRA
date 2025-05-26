import sys
from huggingface_hub import snapshot_download

snapshot_download(repo_id=sys.argv[1])