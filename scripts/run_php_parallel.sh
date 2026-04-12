###################################################################################
# File: run_php_parallel.sh
# File Manager: Kaibiao ZHU
# Description: Parallel runner for PHP single-agent experiments using tmux.
# Usage: bash run_php_parallel.sh
###################################################################################

# ==================== 配置参数 ====================
DATASET="gsm8k"                     # 数据集：math 或 gsm8k
PHP_MODEL="volc_ark"                  # 模型：qwen 或 volc_ark
BASE_PROMPT="complex_cot"                 # 基础提示：standard / cot / complex_cot
MAX_INTERACTIONS=3                 # 每个样本最大交互轮数
TEMPERATURE=0.0                    # 模型温度
CHUNK_SIZE=200                     # 顺序模式下每个分片的样本数量
TOTAL_BATCHES=38                   # 并行批次数（7500 / 200 ≈ 38）
BASE_RESULTS_DIR="results/13_singleself_gsm8k_dou"
LOG_DIR="logs_php"                 # 日志目录
PROJECT_DIR="/home/kaibiaozhu/reagents"  # 实验服务器上的项目路径
# ==================================================

echo "=========================================="
echo "PHP Parallel Runner"
echo "Dataset: $DATASET"
echo "Model: $PHP_MODEL"
echo "Prompt: $BASE_PROMPT"
echo "Chunk Size: $CHUNK_SIZE"
echo "Total Batches: $TOTAL_BATCHES"
echo "Results Dir: $BASE_RESULTS_DIR"
echo "=========================================="

# 创建必要的目录
mkdir -p "$BASE_RESULTS_DIR/chunks" "$LOG_DIR"

for i in $(seq 0 $((TOTAL_BATCHES-1))); do
    START_INDEX=$((i * CHUNK_SIZE))
    SESSION_NAME="php_${i}"

    echo "启动会话 $SESSION_NAME (起始索引: $START_INDEX)"

    tmux new-session -d -s "$SESSION_NAME" "bash -lc '
        source \$(conda info --base)/etc/profile.d/conda.sh && \
        conda activate reagents_env && \
        cd $PROJECT_DIR && \
        python experiments/dataset_test_php.py \\
            --dataset $DATASET \\
            --model $PHP_MODEL \\
            --prompt-type $BASE_PROMPT \\
            --max-interactions $MAX_INTERACTIONS \\
            --temperature $TEMPERATURE \\
            --num-samples $CHUNK_SIZE \\
            --start-index $START_INDEX \\
            --chunk-size $CHUNK_SIZE \\
            --results-dir $BASE_RESULTS_DIR \\
            --block-size $CHUNK_SIZE \\
        |& tee $LOG_DIR/php_${i}.log
    '"
done

echo ""
echo "✅ 已启动 $TOTAL_BATCHES 个并行会话！"
echo ""
echo "管理命令："
echo "  查看所有会话: tmux ls"
echo "  进入会话: tmux attach -t php_0"
echo "  查看日志: tail -f $LOG_DIR/php_0.log"
echo "  杀掉所有会话: tmux ls | awk '/^php_/ {print $1}' | sed 's/:$//' | xargs -r -n1 tmux kill-session -t"
echo ""
