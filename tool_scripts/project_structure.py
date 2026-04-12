import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import datetime

class ProjectStructureGenerator:
    def __init__(self, root_path: str, output_dir: str = "project_structure"):
        self.root_path = Path(root_path)
        self.output_dir = Path(output_dir)
        self.ignore_dirs = {'.git', '__pycache__', '.vscode', '.idea', 'node_modules', 'venv', 'env', '.pytest_cache'}
        self.ignore_files = {'.DS_Store', '.gitignore', '.gitattributes'}
        
    def create_output_directory(self):
        """创建输出目录"""
        self.output_dir.mkdir(exist_ok=True)
        
    def should_ignore(self, path: Path) -> bool:
        """检查是否应该忽略该路径"""
        if path.name in self.ignore_dirs or path.name in self.ignore_files:
            return True
        if path.name.startswith('.'):
            return True
        return False
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """获取文件的详细信息"""
        stat = file_path.stat()
        return {
            'name': file_path.name,
            'size': stat.st_size,
            'modified_time': datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'extension': file_path.suffix.lower()
        }
    
    def scan_directory(self, path: Path, level: int = 0) -> Dict[str, Any]:
        """递归扫描目录结构"""
        if self.should_ignore(path):
            return None
            
        result = {
            'name': path.name,
            'type': 'directory' if path.is_dir() else 'file',
            'level': level,
            'relative_path': str(path.relative_to(self.root_path))
        }
        
        if path.is_file():
            result.update(self.get_file_info(path))
        else:
            result['children'] = []
            try:
                for item in sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
                    child_result = self.scan_directory(item, level + 1)
                    if child_result:
                        result['children'].append(child_result)
            except PermissionError:
                result['permission_error'] = True
                
        return result
    
    def generate_tree_structure(self, data: Dict, prefix: str = "", is_last: bool = True) -> List[str]:
        """生成树形结构的文本表示"""
        lines = []
        
        if data['level'] == 0:
            connector = ""
        else:
            connector = "└── " if is_last else "├── "
        
        line = prefix + connector + data['name']
        if data['type'] == 'file':
            size_kb = data['size'] / 1024
            line += f" ({size_kb:.1f} KB)"
        lines.append(line)
        
        if data['type'] == 'directory' and 'children' in data:
            extension = "    " if is_last else "│   "
            new_prefix = prefix + extension
            
            for i, child in enumerate(data['children']):
                child_is_last = i == len(data['children']) - 1
                lines.extend(self.generate_tree_structure(child, new_prefix, child_is_last))
                
        return lines
    
    def export_to_text(self, data: Dict) -> str:
        """导出为文本格式"""
        output_lines = []
        
        # 头部信息
        output_lines.append("=" * 60)
        output_lines.append(f"项目目录结构报告")
        output_lines.append(f"生成时间: {datetime.datetime.now().isoformat()}")
        output_lines.append(f"项目根目录: {self.root_path}")
        output_lines.append(f"总文件数: {self.count_files(data)}")
        output_lines.append(f"总目录数: {self.count_directories(data)}")
        output_lines.append("=" * 60)
        output_lines.append("")
        
        # 目录树
        tree_lines = self.generate_tree_structure(data)
        output_lines.extend(tree_lines)
        
        # 文件统计
        output_lines.append("")
        output_lines.append("=" * 60)
        output_lines.append("文件类型统计:")
        file_stats = self.get_file_statistics(data)
        for ext, count in sorted(file_stats.items()):
            output_lines.append(f"  {ext or '无扩展名'}: {count} 个文件")
        
        return "\n".join(output_lines)
    
    def export_to_json(self, data: Dict) -> str:
        """导出为JSON格式"""
        report = {
            'metadata': {
                'generated_at': datetime.datetime.now().isoformat(),
                'project_root': str(self.root_path),
                'total_files': self.count_files(data),
                'total_directories': self.count_directories(data)
            },
            'structure': data,
            'statistics': self.get_file_statistics(data)
        }
        return json.dumps(report, indent=2, ensure_ascii=False)
    
    def export_to_markdown(self, data: Dict) -> str:
        """导出为Markdown格式"""
        lines = []
        
        lines.append(f"# 项目目录结构报告")
        lines.append("")
        lines.append(f"- **生成时间**: {datetime.datetime.now().isoformat()}")
        lines.append(f"- **项目根目录**: `{self.root_path}`")
        lines.append(f"- **总文件数**: {self.count_files(data)}")
        lines.append(f"- **总目录数**: {self.count_directories(data)}")
        lines.append("")
        
        lines.append("## 目录结构")
        lines.append("```")
        tree_lines = self.generate_tree_structure(data)
        lines.extend(tree_lines)
        lines.append("```")
        lines.append("")
        
        lines.append("## 文件统计")
        file_stats = self.get_file_statistics(data)
        lines.append("| 文件类型 | 数量 |")
        lines.append("|---------|------|")
        for ext, count in sorted(file_stats.items()):
            lines.append(f"| `{ext or '无扩展名'}` | {count} |")
        
        return "\n".join(lines)
    
    def count_files(self, data: Dict) -> int:
        """统计文件数量"""
        if data['type'] == 'file':
            return 1
        count = 0
        if 'children' in data:
            for child in data['children']:
                count += self.count_files(child)
        return count
    
    def count_directories(self, data: Dict) -> int:
        """统计目录数量"""
        if data['type'] == 'file':
            return 0
        count = 1  # 当前目录
        if 'children' in data:
            for child in data['children']:
                count += self.count_directories(child)
        return count
    
    def get_file_statistics(self, data: Dict) -> Dict[str, int]:
        """获取文件类型统计"""
        stats = {}
        
        def _collect_stats(node):
            if node['type'] == 'file':
                ext = node['extension']
                stats[ext] = stats.get(ext, 0) + 1
            elif 'children' in node:
                for child in node['children']:
                    _collect_stats(child)
        
        _collect_stats(data)
        return stats
    
    def generate_all_formats(self):
        """生成所有格式的输出"""
        self.create_output_directory()
        
        print(f"正在扫描项目目录: {self.root_path}")
        structure_data = self.scan_directory(self.root_path)
        
        if not structure_data:
            print("错误: 无法扫描目录")
            return
        
        # 生成文本格式
        text_output = self.export_to_text(structure_data)
        text_file = self.output_dir / "project_structure.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text_output)
        print(f"✓ 文本格式已保存: {text_file}")
        
        # 生成JSON格式
        json_output = self.export_to_json(structure_data)
        json_file = self.output_dir / "project_structure.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            f.write(json_output)
        print(f"✓ JSON格式已保存: {json_file}")
        
        # 生成Markdown格式
        md_output = self.export_to_markdown(structure_data)
        md_file = self.output_dir / "project_structure.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_output)
        print(f"✓ Markdown格式已保存: {md_file}")
        
        # 在控制台显示摘要
        print(f"\n项目结构摘要:")
        print(f"  总目录数: {self.count_directories(structure_data)}")
        print(f"  总文件数: {self.count_files(structure_data)}")
        
        file_stats = self.get_file_statistics(structure_data)
        print(f"  文件类型分布:")
        for ext, count in sorted(file_stats.items())[:10]:  # 显示前10种类型
            print(f"    {ext or '没有扩展名'}: {count}")

def main():
    parser = argparse.ArgumentParser(description='生成项目目录结构报告')
    parser.add_argument('--root', '-r', default='.', help='项目根目录路径 (默认: 当前目录)')
    parser.add_argument('--output', '-o', default='project_structure', help='输出目录 (默认: project_structure)')
    parser.add_argument('--format', '-f', choices=['all', 'text', 'json', 'markdown'], 
                       default='all', help='输出格式 (默认: all)')
    
    args = parser.parse_args()
    
    generator = ProjectStructureGenerator(args.root, args.output)
    
    try:
        generator.generate_all_formats()
        print(f"\n🎉 项目结构报告生成完成！")
        print(f"   输出位置: {generator.output_dir.absolute()}")
    except Exception as e:
        print(f"❌ 生成过程中出现错误: {e}")

if __name__ == "__main__":
    main()
