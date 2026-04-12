#!/usr/bin/env bash
###################################################################################
# File: run_single_agent_parallel.sh
# File Manager: Kaibiao ZHU
# Description: Single agent模式下大规模并行运行脚本
# Usage: bash run_single_agent_parallel.sh
###################################################################################

# ==================== 配置参数 ====================
DATASET="math"                    # 数据集：math 或 gsm8k
SINGLE_MODEL="qwen"           # 单智能体模型 qwen 或 volc_ark
CHUNK_SIZE=200                    # 每个分片大小
TOTAL_BATCHES=38                  # 总批次数 (根据数据集大小调整)
BASE_RESULTS_DIR="results/9_single_${DATASET}_qwen14b"
# ==================================================

echo "=========================================="
echo "Single Agent Parallel Runner"
echo "Dataset: $DATASET"  
echo "Model: $SINGLE_MODEL"
echo "Chunk Size: $CHUNK_SIZE"
echo "Total Batches: $TOTAL_BATCHES"
echo "Results Dir: $BASE_RESULTS_DIR"
echo "=========================================="

# 创建必要的目录
mkdir -p "$BASE_RESULTS_DIR/chunks" logs_single

# 启动并行 tmux 会话
for i in $(seq 0 $((TOTAL_BATCHES-1))); do
    START_INDEX=$((i * CHUNK_SIZE))
    SESSION_NAME="single_${i}"
    
    echo "启动会话 $SESSION_NAME (起始索引: $START_INDEX)"
    
    tmux new-session -d -s "$SESSION_NAME" "bash -lc '
        source \$(conda info --base)/etc/profile.d/conda.sh && 
        conda activate reagents_env && 
        cd /home/kaibiaozhu/reagents && 
        python experiments/dataset_test_unified_parallel.py \\
            --mode single_agent \\
            --dataset $DATASET \\
            --single-model $SINGLE_MODEL \\
            --start $START_INDEX \\
            --chunk-size $CHUNK_SIZE \\
            --save-results \\
            --results-dir $BASE_RESULTS_DIR \\
        |& tee logs_single/single_${i}.log
    '"
done

echo ""
echo "✅ 已启动 $TOTAL_BATCHES 个并行会话！"
echo ""
echo "管理命令："
echo "  查看所有会话: tmux ls"
echo "  进入会话: tmux attach -t single_0" 
echo "  查看日志: tail -f logs_single/single_0.log"
echo "  杀掉所有会话: tmux ls | awk '/^single_/ {print \$1}' | sed 's/:\$//' | xargs -r -n1 tmux kill-session -t"
echo ""