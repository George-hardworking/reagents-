###################################################################################
# PHP运行脚本
# File Manager: Kaibiao ZHU
###################################################################################

echo "=========================================="
echo "Progressive-Hint Prompting (PHP) 系统"
echo "=========================================="

# 1. 快速测试
echo ""
echo "1. 快速测试 - 单个示例问题"
echo "------------------------------------------"
echo "运行命令: python experiments/test_php_quick.py"
echo ""

# 2. 小规模测试（10个样本）
echo "2. 小规模测试 - 10个样本"
echo "------------------------------------------"
echo "运行命令:"
echo "python experiments/dataset_test_php.py --num-samples 10"
echo ""

# 3. 标准测试（50个样本，随机种子42）
echo "3. 标准测试 - 50个样本（随机种子42）"
echo "------------------------------------------"
echo "运行命令:"
echo "python experiments/dataset_test_php.py \\"
echo "    --num-samples 50 \\"
echo "    --random-seed 42 \\"
echo "    --model qwen \\"
echo "    --prompt-type cot \\"
echo "    --max-interactions 5 \\"
echo "    --temperature 0.0"
echo ""

# 4. 大规模测试（200个样本）
echo "4. 大规模测试 - 200个样本"
echo "------------------------------------------"
echo "运行命令:"
echo "python experiments/dataset_test_php.py --num-samples 200"
echo ""

# 5. 测试不同提示类型
echo "5. 测试不同提示类型"
echo "------------------------------------------"
echo "Standard提示:"
echo "  python experiments/dataset_test_php.py --prompt-type standard --num-samples 50"
echo ""
echo "CoT提示:"
echo "  python experiments/dataset_test_php.py --prompt-type cot --num-samples 50"
echo ""
echo "Complex CoT提示:"
echo "  python experiments/dataset_test_php.py --prompt-type complex_cot --num-samples 50"
echo ""

# 6. 结果分析
echo "6. 结果分析"
echo "------------------------------------------"
echo "合并结果:"
echo "  python tool_scripts/merge_results_chunks.py"
echo ""
echo "统计分析:"
echo "  python tool_scripts/analyze_results.py \\"
echo "      --results_dir results/6_results_self_math_temp0/merged \\"
echo "      --output_dir results/6_results_self_math_temp0/statistic_summary"
echo ""

echo "=========================================="
echo "选择要运行的测试编号，或按Ctrl+C退出"
echo "=========================================="
