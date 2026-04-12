#!/usr/bin/env bash
###################################################################################
# File: run_multi_agent_parallel.sh  
# File Manager: Kaibiao ZHU
# Description: Multi-agent debate模式下大规模并行运行脚本
# Usage: bash run_multi_agent_parallel.sh
###################################################################################

# ==================== 配置参数 ====================
DATASET="math"                    # 数据集：math 或 gsm8k
MODEL1="volc_ark"                 # Agent1 模型 "qwen"or"volc_ark"
MODEL2="volc_ark"                 # Agent2 模型  
CHUNK_SIZE=200                    # 每个分片大小
TOTAL_BATCHES=38                  # 总批次数 (根据数据集大小调整)
BASE_RESULTS_DIR="results/7_multi_math_dou_dou"
#BASE_RESULTS_DIR="results_debate_${DATASET}_${MODEL1}_${MODEL2}_parallel"
# ==================================================

echo "=========================================="
echo "Multi-Agent Debate Parallel Runner"
echo "Dataset: $DATASET"
echo "Model1: $MODEL1" 
echo "Model2: $MODEL2"
echo "Chunk Size: $CHUNK_SIZE"
echo "Total Batches: $TOTAL_BATCHES"
echo "Results Dir: $BASE_RESULTS_DIR"
echo "=========================================="

# 创建必要的目录
mkdir -p "$BASE_RESULTS_DIR/chunks" logs_debate

# 启动并行 tmux 会话
for i in $(seq 0 $((TOTAL_BATCHES-1))); do
    START_INDEX=$((i * CHUNK_SIZE))
    SESSION_NAME="debate_${i}"
    
    echo "启动会话 $SESSION_NAME (起始索引: $START_INDEX)"
    
    tmux new-session -d -s "$SESSION_NAME" "bash -lc '
        source \$(conda info --base)/etc/profile.d/conda.sh && 
        conda activate reagents_env && 
        cd /home/kaibiaozhu/reagents && 
        python experiments/dataset_test_unified_parallel.py \\
            --mode multi_agent \\
            --dataset $DATASET \\
            --model1 $MODEL1 \\
            --model2 $MODEL2 \\
            --start $START_INDEX \\
            --chunk-size $CHUNK_SIZE \\
            --save-results \\
            --results-dir $BASE_RESULTS_DIR \\
        |& tee logs_debate/debate_${i}.log
    '"
done

echo ""
echo "✅ 已启动 $TOTAL_BATCHES 个并行会话！"
echo ""
echo "管理命令："
echo "  查看所有会话: tmux ls"
echo "  进入会话: tmux attach -t debate_0"
echo "  查看日志: tail -f logs_debate/debate_0.log"  
echo "  杀掉所有会话: tmux ls | awk '/^debate_/ {print \$1}' | sed 's/:\$//' | xargs -r -n1 tmux kill-session -t"
echo ""