#!/usr/bin/env bash
###################################################################################
# File: manage_parallel_experiments.sh
# File Manager: Kaibiao ZHU
# Description: 管理并行实验的通用脚本
###################################################################################

show_usage() {
    echo "用法: bash manage_parallel_experiments.sh [COMMAND]"
    echo ""
    echo "命令:"
    echo "  status      - 查看所有运行中的实验会话"
    echo "  logs        - 查看最新日志输出"
    echo "  kill-single - 杀掉所有 single agent 会话"
    echo "  kill-debate - 杀掉所有 multi-agent debate 会话" 
    echo "  kill-all    - 杀掉所有实验会话"
    echo "  results     - 检查结果文件数量"
    echo "  monitor     - 实时监控第一个会话的日志"
}

case "$1" in
    "status")
        echo "=== 运行中的实验会话 ==="
        tmux ls | grep -E "^(single_|debate_)" || echo "没有运行中的实验会话"
        ;;
    
    "logs")
        echo "=== 最新日志输出 ==="
        echo "Single Agent 日志:"
        if [ -d "logs_single" ]; then
            find logs_single -name "*.log" -exec tail -n 3 {} + 2>/dev/null || echo "没有找到 single agent 日志"
        fi
        echo ""
        echo "Multi-Agent Debate 日志:"
        if [ -d "logs_debate" ]; then
            find logs_debate -name "*.log" -exec tail -n 3 {} + 2>/dev/null || echo "没有找到 debate 日志"
        fi
        ;;
    
    "kill-single")
        echo "杀掉所有 single agent 会话..."
        tmux ls | awk '/^single_/ {print $1}' | sed 's/:$//' | xargs -r -n1 tmux kill-session -t
        echo "完成!"
        ;;
    
    "kill-debate")
        echo "杀掉所有 multi-agent debate 会话..."
        tmux ls | awk '/^debate_/ {print $1}' | sed 's/:$//' | xargs -r -n1 tmux kill-session -t
        echo "完成!"
        ;;
    
    "kill-all")
        echo "杀掉所有实验会话..."
        tmux ls | awk '/^(single_|debate_)/ {print $1}' | sed 's/:$//' | xargs -r -n1 tmux kill-session -t
        echo "完成!"
        ;;
    
    "results")
        echo "=== 结果文件统计 ==="
        for dir in results_*; do
            if [ -d "$dir" ]; then
                count=$(find "$dir" -name "block_results_*.json" 2>/dev/null | wc -l)
                echo "$dir: $count 个结果文件"
            fi
        done
        ;;
    
    "monitor")
        echo "实时监控日志 (Ctrl+C 退出):"
        if [ -f "logs_single/single_0.log" ]; then
            echo "监控 single_0 日志..."
            tail -f logs_single/single_0.log
        elif [ -f "logs_debate/debate_0.log" ]; then
            echo "监控 debate_0 日志..." 
            tail -f logs_debate/debate_0.log
        else
            echo "没有找到可监控的日志文件"
        fi
        ;;
    
    *)
        show_usage
        ;;
esac