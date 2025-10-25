from modelscope.hub.snapshot_download import snapshot_download

# 下载模型到本地（会自动从国内镜像拉取）
model_dir = snapshot_download(
    "deepseek-ai/deepseek-llm-7b-chat",
    cache_dir="DS_model"  # 可选，指定保存目录
)
print("模型下载路径：", model_dir)