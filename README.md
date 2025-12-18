# maml-agent

export APPWORLD_ROOT="/workspace/maml/maml-agent"
export HF_TOKEN=
vllm serve microsoft/Phi-3-mini-4k-instruct --port 8000

microsoft/Phi-3-mini-128k-instruct

vllm serve microsoft/Phi-3-mini-128k-instruct \
  --port 8000 \
  --max-model-len 32768 # gpu capacity dependent


 when install appworld always have to upgrade packages:
 pip install "click<8.2" --force-reinstall
