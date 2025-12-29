"""
LaTeX 编译错误自动修复模块
通过解析编译日志，提取错误信息，使用 LLM 自动修复 LaTeX 文件
"""
import re
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

from src.configs.logger import get_logger
from src.models.LLM import ChatAgent
from src.configs.config import ADVANCED_CHATAGENT_MODEL
from src.modules.utils import load_file_as_string, save_result

logger = get_logger("src.modules.latex_handler.LatexErrorFixer")


class LaTeXErrorFixer:
    """LaTeX 编译错误自动修复器"""
    
    def __init__(self, task_id: str, latex_dir: Path):
        self.task_id = task_id
        self.latex_dir = latex_dir
        self.survey_tex_path = latex_dir / "survey.tex"
        self.compile_log_path = latex_dir / "compile.log"
        self.chat_agent = ChatAgent()
        self.max_fix_attempts = 3  # 最大修复尝试次数
        
    def parse_compile_errors(self) -> List[Tuple[str, int, str]]:
        """
        解析编译日志，提取错误信息
        返回: [(错误类型, 行号, 错误描述), ...]
        """
        if not self.compile_log_path.exists():
            return []
        
        errors = []
        log_content = load_file_as_string(self.compile_log_path)
        lines = log_content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # 匹配 LaTeX 错误格式: ! LaTeX Error: ...
            if '! LaTeX Error:' in line:
                error_type = line.split('! LaTeX Error:')[1].strip()
                # 查找行号: l.123
                line_num = None
                error_desc = [error_type]
                
                # 向前查找行号
                for j in range(max(0, i-5), i):
                    match = re.search(r'l\.(\d+)', lines[j])
                    if match:
                        line_num = int(match.group(1))
                        break
                
                # 收集错误描述（后续几行）
                for j in range(i+1, min(i+5, len(lines))):
                    if lines[j].strip() and not lines[j].strip().startswith('!'):
                        error_desc.append(lines[j].strip())
                    elif lines[j].strip().startswith('!'):
                        break
                
                errors.append((
                    error_type,
                    line_num,
                    ' '.join(error_desc)
                ))
            
            # 匹配其他严重错误
            elif '! Emergency stop' in line or 'Fatal error' in line or 'Fatal error occurred' in line:
                errors.append((
                    "Fatal Error",
                    None,
                    "Emergency stop or fatal error occurred"
                ))
            
            # 匹配其他错误格式
            elif line.strip().startswith('!') and ('Error' in line or 'error' in line):
                error_type = line.strip()
                errors.append((
                    "LaTeX Error",
                    None,
                    error_type
                ))
            
            # 检查是否有编译失败的提示
            elif 'gave an error' in line.lower() or 'error in previous invocation' in line.lower():
                errors.append((
                    "Previous Compilation Error",
                    None,
                    line.strip()
                ))
            
            i += 1
        
        return errors
    
    def extract_error_context(self, line_num: Optional[int], context_lines: int = 5) -> str:
        """提取错误行周围的上下文"""
        if not self.survey_tex_path.exists():
            return ""
        
        content = load_file_as_string(self.survey_tex_path)
        lines = content.split('\n')
        
        if line_num is None:
            return content[:500]  # 返回前500个字符
        
        start = max(0, line_num - context_lines - 1)
        end = min(len(lines), line_num + context_lines)
        
        context = []
        for i in range(start, end):
            prefix = ">>> " if i == line_num - 1 else "    "
            context.append(f"{prefix}Line {i+1}: {lines[i]}")
        
        return '\n'.join(context)
    
    def extract_error_regions(self, errors: List[Tuple[str, int, str]], tex_content: str, context_lines: int = 50) -> List[tuple]:
        """
        根据错误位置提取相关代码区域
        返回: [(start_line, end_line, error_info), ...] 列表，按行号排序并合并重叠区域
        """
        lines = tex_content.split('\n')
        regions = []
        
        for error_type, line_num, desc in errors:
            if line_num is None:
                # 如果错误没有明确的行号，跳过（可能是全局错误）
                continue
            
            # 计算区域范围（包含上下文）
            start_line = max(0, line_num - context_lines - 1)
            end_line = min(len(lines), line_num + context_lines)
            
            regions.append((start_line, end_line, (error_type, line_num, desc)))
        
        # 如果没有找到有行号的错误，返回空列表
        if not regions:
            return []
        
        # 按起始行号排序
        regions.sort(key=lambda x: x[0])
        
        # 合并重叠的区域
        merged_regions = []
        for start, end, error_info in regions:
            if not merged_regions:
                merged_regions.append((start, end, [error_info]))
            else:
                last_start, last_end, last_errors = merged_regions[-1]
                # 如果当前区域与上一个区域重叠或相邻（间隔小于 context_lines），合并
                if start <= last_end + context_lines:
                    merged_regions[-1] = (last_start, max(last_end, end), last_errors + [error_info])
                else:
                    merged_regions.append((start, end, [error_info]))
        
        return merged_regions
    
    def create_fix_prompt_for_region(self, region_content: str, errors_in_region: List[Tuple[str, int, str]], full_file_info: str) -> str:
        """为特定区域创建修复提示词"""
        error_summary = []
        for i, (error_type, line_num, desc) in enumerate(errors_in_region, 1):
            error_summary.append(f"""
Error {i}:
- Type: {error_type}
- Line: {line_num if line_num else 'Unknown'}
- Description: {desc[:200]}
""")
        
        prompt = f"""You are a LaTeX expert. The following LaTeX code snippet contains compilation errors. Please fix ONLY the errors in this snippet and return the fixed code.

Full File Context (for reference):
{full_file_info[:2000]}

Error List in This Region:
{''.join(error_summary)}

Code Snippet to Fix:
{region_content}

Please fix according to the following requirements:
1. **Return ONLY the fixed code snippet**, not the entire file
2. Fix all compilation errors in this snippet
3. Keep the code structure and content unchanged, only fix syntax errors
4. If the error is a package conflict (e.g., natbib), remove the duplicate package declaration
5. If the error is an unclosed environment (e.g., table), add the missing \\end{{table}}
6. If the error is a missing image path (e.g., path_to_image), comment out the image
7. If the error is a citation format issue, fix the citation format (e.g., remove special characters like !, %)

**IMPORTANT: Return ONLY the fixed code snippet, maintaining the same line structure and indentation.**"""
        
        return prompt
    
    def merge_fixed_regions(self, original_content: str, fixed_regions: List[tuple]) -> str:
        """
        将修复后的区域合并回原文件
        fixed_regions: [(start_line, end_line, fixed_content), ...]
        """
        lines = original_content.split('\n')
        # 按起始行号倒序排序，从后往前替换，避免行号变化
        fixed_regions.sort(key=lambda x: x[0], reverse=True)
        
        for start_line, end_line, fixed_content in fixed_regions:
            # 将修复后的内容按行分割
            fixed_lines = fixed_content.split('\n')
            # 替换原文件中的对应区域
            lines[start_line:end_line] = fixed_lines
        
        return '\n'.join(lines)
    
    def create_fix_prompt(self, errors: List[Tuple[str, int, str]], tex_content: str) -> str:
        """创建修复提示词（保留原方法作为备用）"""
        error_summary = []
        for i, (error_type, line_num, desc) in enumerate(errors[:10], 1):
            context = self.extract_error_context(line_num)
            error_summary.append(f"""
Error {i}:
- Type: {error_type}
- Line: {line_num if line_num else 'Unknown'}
- Description: {desc[:200]}
- Context:
{context}
""")
        
        # 发送完整的文件内容
        max_chars = 50000
        if len(tex_content) <= max_chars:
            content_preview = tex_content
            logger.debug(f"发送完整文件内容: {len(tex_content)} 字符")
        else:
            front_chars = 30000
            back_chars = 20000
            content_preview = (
                tex_content[:front_chars] + 
                "\n\n[... The file content has been truncated, but please fix all errors. ...]\n\n" + 
                tex_content[-back_chars:]
            )
            logger.debug(f"文件过长，发送部分内容: 前{front_chars} + 后{back_chars} = {front_chars + back_chars} 字符")
        
        prompt = f"""You are a LaTeX expert. The following LaTeX file encountered compilation errors. Please fix these errors and return the complete, repaired file.

Compilation Error List:
{''.join(error_summary)}

Current LaTeX File Content:
{content_preview}

Please fix according to the following requirements:
1. **You MUST return the complete fixed LaTeX file**, not just the modified parts
2. Fix all compilation errors to ensure the file can compile successfully
3. Keep the LaTeX document structure and content unchanged
4. Only fix syntax errors, do not change the document semantics
5. If the error is a package conflict (e.g., natbib), remove the duplicate package declaration
6. If the error is an unclosed environment (e.g., table), add the missing \\end{{table}}
7. If the error is a missing image path (e.g., path_to_image), comment out the image
8. If the error is a citation format issue, fix the citation format

**IMPORTANT: Please return the complete fixed LaTeX file content, from \\documentclass to \\end{{document}}.**"""
        
        return prompt
    
    def apply_rule_based_fixes(self, content: str) -> str:
        """应用基于规则的快速修复（用于常见错误）"""
        fixed = content
        import re
        
        # 修复 natbib 冲突：移除重复的 \usepackage[numbers]{natbib}
        # 因为 neurips_2024.sty 已经加载了 natbib
        if r'\usepackage[numbers]{natbib}' in fixed:
            logger.info("应用规则修复：移除重复的 natbib 包声明")
            fixed = fixed.replace(r'\usepackage[numbers]{natbib}', '% Removed: neurips_2024.sty already loads natbib')
        
        # 修复图片路径占位符
        if 'path_to_image' in fixed:
            logger.info("应用规则修复：注释掉 path_to_image 图片")
            # 注释掉包含 path_to_image 的 includegraphics
            pattern = r'\\includegraphics.*?\{path_to_image\}'
            fixed = re.sub(pattern, lambda m: '% ' + m.group(0) + ' % Fixed: placeholder image', fixed)
        
        # 修复引用格式错误：处理包含特殊字符的引用（如 \cite{123Go!Poli0} 或 \cite{99%ofDistr1}）
        # 查找 \cite{...} 中包含特殊字符的引用
        cite_pattern = r'\\cite\{([^}]*)\}'
        def fix_cite_key(match):
            cite_key = match.group(1)
            # 如果包含特殊字符，转义或移除
            if any(char in cite_key for char in ['!', '%', '@', '#', '$', '^', '&', '*']):
                logger.info(f"应用规则修复：修复引用格式中的特殊字符: {cite_key[:50]}")
                # 移除或转义特殊字符
                fixed_key = cite_key.replace('!', '').replace('%', '').replace('@', '').replace('#', '').replace('$', '').replace('^', '').replace('&', '').replace('*', '')
                return f'\\cite{{{fixed_key}}}'
            return match.group(0)
        
        fixed = re.sub(cite_pattern, fix_cite_key, fixed)
        
        # 修复未闭合的表格：检查是否有 \begin{table} 但没有对应的 \end{table}
        # 统计 begin{table} 和 end{table} 的数量
        begin_tables = len(re.findall(r'\\begin\{table\}', fixed))
        end_tables = len(re.findall(r'\\end\{table\}', fixed))
        if begin_tables > end_tables:
            logger.info(f"应用规则修复：发现 {begin_tables - end_tables} 个未闭合的表格")
            # 在 \end{document} 之前添加缺失的 \end{table}
            missing_ends = begin_tables - end_tables
            fixed = fixed.replace(r'\end{document}', 
                                 '\n' + '\\end{table}\n' * missing_ends + r'\end{document}')
        
        return fixed
    
    def fix_latex_errors_with_llm(self, errors: List[Tuple[str, int, str]], original_content: str) -> tuple[bool, str]:
        """
        使用 LLM 修复 LaTeX 错误（智能区域修复策略）
        策略：根据错误位置提取相关代码区域，分别修复后合并回原文件
        返回: (是否成功, 修复后的内容)
        """
        # 提取错误区域
        error_regions = self.extract_error_regions(errors, original_content)
        
        if not error_regions:
            # 如果没有找到有行号的错误，使用完整文件修复（备用策略）
            logger.warning("未找到有明确行号的错误，使用完整文件修复策略")
            return self._fix_full_file_with_llm(errors, original_content)
        
        logger.info(f"找到 {len(error_regions)} 个错误区域，使用智能区域修复策略")
        
        lines = original_content.split('\n')
        fixed_regions = []
        
        # 为每个区域生成文件信息（用于上下文）
        file_info = f"Total lines: {len(lines)}\nDocument structure: {original_content[:500]}..."
        
        # 逐个修复每个区域
        for region_idx, (start_line, end_line, errors_in_region) in enumerate(error_regions, 1):
            logger.info(f"修复区域 {region_idx}/{len(error_regions)}: 行 {start_line+1}-{end_line+1} ({len(errors_in_region)} 个错误)")
            
            # 提取区域内容
            region_lines = lines[start_line:end_line]
            region_content = '\n'.join(region_lines)
            
            # 创建修复提示词
            prompt = self.create_fix_prompt_for_region(region_content, errors_in_region, file_info)
            
            try:
                # 调用 LLM 修复
                logger.debug(f"调用 LLM 修复区域 {region_idx}...")
                fixed_region = self.chat_agent.remote_chat(
                    prompt,
                    model=ADVANCED_CHATAGENT_MODEL
                )
                
                # 清理返回内容
                fixed_region = fixed_region.strip()
                
                # 提取代码块
                if "```" in fixed_region:
                    import re
                    code_blocks = re.findall(r'```(?:latex)?\s*\n(.*?)```', fixed_region, re.DOTALL)
                    if code_blocks:
                        fixed_region = code_blocks[0].strip()
                    else:
                        fixed_region = re.sub(r'^```(?:latex)?\s*\n?', '', fixed_region)
                        fixed_region = re.sub(r'\n?```\s*$', '', fixed_region)
                        fixed_region = fixed_region.strip()
                
                # 验证修复后的区域长度（应该与原区域相近）
                if len(fixed_region) < len(region_content) * 0.3:
                    logger.warning(f"区域 {region_idx} 修复后内容过短，可能不完整")
                    logger.warning(f"原区域长度: {len(region_content)}, 修复后长度: {len(fixed_region)}")
                    # 如果修复失败，保留原内容
                    fixed_region = region_content
                elif len(fixed_region) > len(region_content) * 2:
                    logger.warning(f"区域 {region_idx} 修复后内容过长，可能包含额外内容")
                    # 尝试截取到合理长度
                    fixed_lines = fixed_region.split('\n')
                    if len(fixed_lines) > (end_line - start_line) * 2:
                        fixed_region = '\n'.join(fixed_lines[:end_line - start_line + 10])
                
                fixed_regions.append((start_line, end_line, fixed_region))
                logger.info(f"区域 {region_idx} 修复完成")
                
            except Exception as e:
                logger.error(f"修复区域 {region_idx} 时发生异常: {e}")
                # 修复失败，保留原内容
                fixed_regions.append((start_line, end_line, region_content))
        
        # 合并修复后的区域回原文件
        if fixed_regions:
            fixed_content = self.merge_fixed_regions(original_content, fixed_regions)
            logger.info(f"已合并 {len(fixed_regions)} 个修复区域回原文件")
            return True, fixed_content
        else:
            return False, original_content
    
    def _fix_full_file_with_llm(self, errors: List[Tuple[str, int, str]], original_content: str) -> tuple[bool, str]:
        """
        使用完整文件修复策略（备用方法）
        返回: (是否成功, 修复后的内容)
        """
        prompt = self.create_fix_prompt(errors, original_content)
        
        try:
            logger.info("正在调用 LLM 修复完整文件...")
            fixed_content = self.chat_agent.remote_chat(
                prompt,
                model=ADVANCED_CHATAGENT_MODEL
            )
            
            # 清理 LLM 返回的内容
            fixed_content = fixed_content.strip()
            
            # 尝试提取 LaTeX 代码块
            if "```" in fixed_content:
                import re
                code_blocks = re.findall(r'```(?:latex)?\s*\n(.*?)```', fixed_content, re.DOTALL)
                if code_blocks:
                    fixed_content = code_blocks[0].strip()
                else:
                    fixed_content = re.sub(r'^```(?:latex)?\s*\n?', '', fixed_content)
                    fixed_content = re.sub(r'\n?```\s*$', '', fixed_content)
                    fixed_content = fixed_content.strip()
            
            # 检查返回内容的完整性
            if len(fixed_content) < len(original_content) * 0.3:
                logger.warning("LLM 返回的内容过短，可能只是部分修复")
                logger.warning(f"原文件长度: {len(original_content)}, 返回长度: {len(fixed_content)}")
                return False, fixed_content
            elif len(fixed_content) < len(original_content) * 0.7:
                logger.warning("LLM 返回的内容可能不完整，但尝试使用")
            
            return True, fixed_content
            
        except Exception as e:
            logger.error(f"LLM 修复时发生异常: {e}")
            return False, ""
    
    def fix_latex_errors(self, use_llm_first: bool = True) -> bool:
        """
        尝试修复 LaTeX 错误
        参数:
            use_llm_first: 是否优先使用 LLM 修复（默认 True）
        返回: 是否修复成功
        """
        errors = self.parse_compile_errors()
        
        if not errors:
            logger.info("未发现编译错误")
            return True
        
        logger.warning(f"发现 {len(errors)} 个编译错误，开始自动修复...")
        
        # 显示错误摘要
        for error_type, line_num, desc in errors[:5]:
            logger.warning(f"错误: {error_type} (行 {line_num}): {desc[:100]}")
        
        # 读取当前 LaTeX 文件
        if not self.survey_tex_path.exists():
            logger.error(f"LaTeX 文件不存在: {self.survey_tex_path}")
            return False
        
        original_content = load_file_as_string(self.survey_tex_path)
        
        # 优先使用 LLM 修复
        if use_llm_first:
            logger.info("优先尝试使用 LLM 修复...")
            success, fixed_content = self.fix_latex_errors_with_llm(errors, original_content)
            
            if success:
                # 保存 LLM 修复后的文件
                save_result(fixed_content, self.survey_tex_path)
                logger.info(f"已保存 LLM 修复后的 LaTeX 文件: {self.survey_tex_path}")
                logger.info(f"修复后文件长度: {len(fixed_content)} (原文件: {len(original_content)})")
                return True
            else:
                logger.warning("LLM 修复失败或返回内容不完整")
        
        # LLM 修复失败，使用基于规则的修复作为备用
        logger.info("LLM 修复无效，尝试基于规则的快速修复...")
        rule_fixed_content = self.apply_rule_based_fixes(original_content)
        
        if rule_fixed_content != original_content:
            save_result(rule_fixed_content, self.survey_tex_path)
            logger.info("已应用基于规则的修复")
            return True
        else:
            logger.warning("基于规则的修复也未产生变化")
            return False
    
    def compile_and_fix(self) -> bool:
        """
        编译 LaTeX 文件，如果失败则尝试修复
        策略：先尝试 LLM 修复（最多3次），如果都失败，再使用基于规则的修复
        返回: 是否编译成功
        """
        llm_attempts = 0
        max_llm_attempts = 3
        
        for attempt in range(self.max_fix_attempts):
            logger.info(f"编译尝试 {attempt + 1}/{self.max_fix_attempts}")
            
            # 编译（强制重新编译，清除之前的缓存）
            with open(self.compile_log_path, "w") as log_file:
                # 先清理之前的编译产物
                subprocess.run("latexmk -C", shell=True, cwd=self.latex_dir, 
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # 强制重新编译
                result = subprocess.run(
                    "latexmk -pdf -interaction=nonstopmode -f survey.tex",
                    shell=True,
                    cwd=self.latex_dir,
                    stdout=log_file,
                    stderr=log_file,
                )
            
            # 检查是否生成了 PDF
            pdf_path = self.latex_dir / "survey.pdf"
            if pdf_path.exists():
                logger.info("LaTeX 编译成功！")
                return True
            
            # 如果编译失败且还有尝试次数，尝试修复
            if attempt < self.max_fix_attempts - 1:
                logger.warning("编译失败，尝试自动修复...")
                errors = self.parse_compile_errors()
                
                # 决定修复策略
                use_llm = llm_attempts < max_llm_attempts
                
                if not errors:
                    logger.warning("未发现明确的编译错误，但 PDF 未生成")
                    if use_llm:
                        logger.info("尝试 LLM 修复...")
                        # 创建一个通用错误列表用于 LLM
                        errors = [("PDF Not Generated", None, "PDF file was not generated despite compilation")]
                
                if errors:
                    if use_llm:
                        logger.info(f"使用 LLM 修复（尝试 {llm_attempts + 1}/{max_llm_attempts}）...")
                        if self.fix_latex_errors(use_llm_first=True):
                            llm_attempts += 1
                            logger.info("LLM 修复完成，重新编译...")
                        else:
                            llm_attempts += 1
                            logger.warning(f"LLM 修复失败（尝试 {llm_attempts}/{max_llm_attempts}）")
                            if llm_attempts >= max_llm_attempts:
                                logger.info("LLM 修复已达到最大尝试次数，切换到基于规则的修复...")
                                if not self.fix_latex_errors(use_llm_first=False):
                                    logger.error("基于规则的修复也失败")
                                    return False
                            else:
                                logger.info("继续尝试 LLM 修复...")
                    else:
                        logger.info("LLM 修复已达到最大尝试次数，使用基于规则的修复...")
                        if not self.fix_latex_errors(use_llm_first=False):
                            logger.error("基于规则的修复失败")
                            return False
                else:
                    # 没有明确的错误，尝试基于规则的修复
                    logger.info("尝试基于规则的修复...")
                    original_content = load_file_as_string(self.survey_tex_path)
                    rule_fixed_content = self.apply_rule_based_fixes(original_content)
                    if rule_fixed_content != original_content:
                        save_result(rule_fixed_content, self.survey_tex_path)
                        logger.info("已应用基于规则的修复")
                    else:
                        logger.warning("基于规则的修复也未产生变化")
                
                logger.info("已修复，重新编译...")
            else:
                logger.error("达到最大修复尝试次数，编译仍然失败")
                return False
        
        return False

