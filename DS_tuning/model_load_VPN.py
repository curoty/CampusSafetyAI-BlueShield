from huggingface_hub import snapshot_download

# 指定模型名称
model_id = "deepseek-ai/deepseek-llm-7b-chat"

# 下载到默认缓存目录（C:\Users\<用户名>\.cache\huggingface\hub）
repo_dir = snapshot_download(repo_id=model_id, local_dir_use_symlinks=False)

print("✅ 模型已下载到:", repo_dir)
