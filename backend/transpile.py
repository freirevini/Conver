#!/usr/bin/env python
"""
KNIME to Python Transpiler - Standalone Minimal Version

NÃO requer instalação de dependências externas.
Usa apenas bibliotecas padrão do Python.

Usage:
    python transpile.py arquivo.knwf
    
Output:
    - arquivo.py     (código Python gerado)
    - arquivo_log.md (log de diagnóstico)
"""
import sys
import os
import zipfile
import tempfile
import shutil
import json
import re
import html
from datetime import datetime
from pathlib import Path
from collections import Counter

# LLM Fallback for unknown nodes (optional, graceful degradation)
try:
    from llm_fallback import llm_fallback, get_llm_client
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    def llm_fallback(factory, node_name, settings=None):
        """Fallback stub when LLM module not available."""
        simple = factory.split('.')[-1].replace('NodeFactory', '')
        return f'pass  # TODO: Implement {simple}', False

# LLM translator for StringManipulation and Java Snippet expressions (optional)
try:
    from llm_string_translator import translate_cached as llm_translate_string
    from llm_string_translator import translate_java_snippet as llm_translate_java
    LLM_STRING_AVAILABLE = True
except ImportError:
    LLM_STRING_AVAILABLE = False
    def llm_translate_string(expr, target, df_var='df'):
        return ''
    def llm_translate_java(java_code, in_cols, out_cols, df_var='df'):
        return ''


def _extract_column_list(content):
    """Extract column names from KNIME XML included_names/InclList/filter config.
    
    Handles both flat and nested patterns:
    - Flat:   <config key="included_names">...</config>
    - Nested: <config key="col_select"><config key="included_names">...</config></config>
    """
    # 1. Try nested col_select > included_names (OldToNewTime, NewToOldTime)
    cs_start = content.find('"col_select"')
    if cs_start >= 0:
        # Find included_names within the col_select section
        incl_start = content.find('"included_names"', cs_start)
        if incl_start >= 0 and incl_start < cs_start + 2000:
            # Grab content up to next </config>
            incl_end = content.find('</config>', incl_start)
            if incl_end >= 0:
                incl_section = content[incl_start:incl_end]
                cols = re.findall(r'<entry\s+key="\d+"\s+type="xstring"\s+value="([^"]+)"', incl_section)
                if cols:
                    return cols
    # 2. Flat keys
    for key in ['included_names', 'InclList', 'column_filter', 'filter']:
        section = re.search(rf'<config\s+key="{key}"[^>]*>(.*?)</config>', content, re.DOTALL)
        if section:
            cols = re.findall(r'<entry\s+key="\d+"\s+type="xstring"\s+value="([^"]+)"', section.group(1))
            if cols:
                return cols
    return []


class TranspilerLog:
    """Collects diagnostic information."""
    
    def __init__(self, input_path):
        self.input_path = input_path
        self.start_time = datetime.now()
        self.extraction_method = ""
        self.nodes_found = []
        self.template_matches = []
        self.fallback_nodes = []
        self.errors = []
        self.warnings = []
        self._unique_warnings = set()
        self.factory_counts = Counter()
    
    def add_node(self, name, factory, matched, template_name=""):
        self.nodes_found.append({
            'name': name,
            'factory': factory,
            'matched': matched,
            'template': template_name
        })
        
        simple_factory = factory.split('.')[-1] if factory else 'Unknown'
        self.factory_counts[simple_factory] += 1
        
        if matched:
            self.template_matches.append({'name': name, 'factory': simple_factory, 'template': template_name})
        else:
            self.fallback_nodes.append({'name': name, 'factory': simple_factory})
    
    def add_error(self, msg):
        self.errors.append(msg)
    
    def add_warning(self, msg):
        if msg not in self._unique_warnings:
            self._unique_warnings.add(msg)
            self.warnings.append(msg)
    
    def generate_markdown(self):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        total = len(self.nodes_found)
        matched = len(self.template_matches)
        fallback = len(self.fallback_nodes)
        coverage = (matched / total * 100) if total > 0 else 0
        
        lines = [
            "# Transpilation Log",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Input | `{Path(self.input_path).name}` |",
            f"| Timestamp | {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} |",
            f"| Duration | {duration:.2f}s |",
            f"| Method | {self.extraction_method} |",
            f"| Total Nodes | {total} |",
            f"| Template Matches | {matched} |",
            f"| Fallback | {fallback} |",
            f"| Coverage | {coverage:.1f}% |",
            "",
        ]
        
        if self.errors:
            lines.extend(["## Errors", ""])
            for err in self.errors:
                lines.append(f"- `{err}`")
            lines.append("")
        
        if self.warnings:
            lines.extend(["## Warnings", ""])
            for warn in self.warnings:
                lines.append(f"- {warn}")
            lines.append("")
        
        lines.extend([
            "## Factory Distribution (Top 20)",
            "",
            "| Factory | Count |",
            "|---------|-------|",
        ])
        for factory, count in self.factory_counts.most_common(20):
            lines.append(f"| {factory} | {count} |")
        lines.append("")
        
        # ── Translation Method Summary ──
        method_counts = Counter()
        method_categories = {
            'Dedicado': [],
            'Determinístico': [],
            'LLM': [],
            'Template': [],
            'Injeção Direta': [],
            'Fallback': [],
        }
        for node in self.template_matches:
            tpl = node.get('template', '')
            if 'Deterministic' in tpl or 'deterministic' in tpl:
                cat = 'Determinístico'
            elif 'LLM' in tpl or 'llm' in tpl:
                cat = 'LLM'
            elif 'Injected' in tpl or 'injected' in tpl:
                cat = 'Injeção Direta'
            elif 'Extracted' in tpl:
                cat = 'Template'
            elif tpl and tpl not in ('template', ''):
                cat = 'Dedicado'
            else:
                cat = 'Template'
            method_counts[cat] += 1
            method_categories[cat].append(node)
        for node in self.fallback_nodes:
            method_counts['Fallback'] += 1
            method_categories['Fallback'].append(node)
        
        lines.extend([
            "## Translation Method Summary",
            "",
            "| Método | Nodes | % | Descrição |",
            "|--------|-------|---|-----------|",
        ])
        method_descriptions = {
            'Dedicado': 'Extração de parâmetros + geração de código específico',
            'Determinístico': 'Regex/pattern matching sem LLM (StringManip, Java)',
            'LLM': 'Tradução via Gemini 2.5 Pro (Vertex AI)',
            'Injeção Direta': 'Código Python extraído e injetado (Python2Script)',
            'Template': 'Template pré-definido no TEMPLATE_MAP',
            'Fallback': 'Nó sem mapeamento — gerado como comentário',
        }
        total_nodes = len(self.template_matches) + len(self.fallback_nodes)
        for cat in ['Dedicado', 'Determinístico', 'Injeção Direta', 'LLM', 'Template', 'Fallback']:
            count = method_counts.get(cat, 0)
            if count > 0:
                pct = f"{count / total_nodes * 100:.1f}%"
                desc = method_descriptions.get(cat, '')
                lines.append(f"| {cat} | {count} | {pct} | {desc} |")
        lines.append("")
        
        # ── Special Nodes Detail ──
        special_types = {
            'StringManip_Deterministic': ('StringManipulation', 'Expressão KNIME → .apply(lambda)'),
            'StringManip_LLM': ('StringManipulation (LLM)', 'Expressão KNIME → LLM → pandas'),
            'StringManip_Extracted': ('StringManipulation', 'Expressão extraída (template)'),
            'JavaSnippet_Deterministic': ('Java Snippet', 'if/else Java → np.where()'),
            'JavaEditVar_Deterministic': ('Java Edit Variable', 'Atribuição simples → flow var'),
            'Java_LLM': ('Java Snippet (LLM)', 'Código Java → LLM → pandas'),
            'PythonScript_Injected': ('Python2Script', 'Código Python extraído e injetado'),
            'PythonScript_Extracted': ('Python2Script', 'Código Python (AST falhou)'),
        }
        
        special_nodes = [
            n for n in self.template_matches
            if any(s in n.get('template', '') for s in special_types)
        ]
        
        if special_nodes:
            lines.extend([
                "## Special Nodes Detail",
                "",
                "| Node | Factory | Método | Tradução |",
                "|------|---------|--------|----------|",
            ])
            for node in special_nodes:
                tpl = node.get('template', '')
                name = node.get('name', '')[:35]
                factory = node.get('factory', '')
                for key, (tipo, desc) in special_types.items():
                    if key in tpl:
                        lines.append(f"| {name} | {tipo} | {key} | {desc} |")
                        break
            lines.append("")
        
        # ── Full Node List ──
        lines.extend([
            "## Full Node List",
            "",
            "| # | Node | Factory | Método |",
            "|---|------|---------|--------|",
        ])
        
        all_nodes = []
        for node in self.template_matches:
            tpl = node.get('template', '')
            if 'Deterministic' in tpl or 'deterministic' in tpl:
                method_label = f"Determinístico ({tpl})"
            elif 'LLM' in tpl:
                method_label = f"LLM ({tpl})"
            elif 'Injected' in tpl:
                method_label = "Injeção Direta"
            elif tpl and tpl not in ('template', ''):
                method_label = f"Dedicado ({tpl})"
            else:
                method_label = "Template"
            all_nodes.append((node.get('name', ''), node.get('factory', ''), method_label))
        
        for node in self.fallback_nodes:
            all_nodes.append((node.get('name', ''), node.get('factory', ''), "Fallback"))
        
        for idx, (name, factory, method) in enumerate(all_nodes, 1):
            short_name = name[:30]
            lines.append(f"| {idx} | {short_name} | {factory} | {method} |")
        lines.append("")
        
        if self.fallback_nodes:
            lines.extend([
                "## Fallback Nodes (No Template)",
                "",
                "| Node | Factory |",
                "|------|---------|",
            ])
            fallback_by_factory = {}
            for item in self.fallback_nodes:
                factory = item['factory']
                if factory not in fallback_by_factory:
                    fallback_by_factory[factory] = []
                fallback_by_factory[factory].append(item['name'])
            
            for factory, names in sorted(fallback_by_factory.items(), key=lambda x: -len(x[1])):
                sample = names[0][:25] if names else ""
                if len(names) > 1:
                    lines.append(f"| {sample}... ({len(names)}x) | {factory} |")
                else:
                    lines.append(f"| {sample} | {factory} |")
            lines.append("")
        
        lines.extend([
            "## All Unique Factories",
            "",
            "```",
        ])
        for factory in sorted(set(n.get('factory', '') for n in self.nodes_found if n.get('factory'))):
            lines.append(factory.split('.')[-1])
        lines.extend(["```", ""])
        
        return '\n'.join(lines)


def decode_knime_escapes(text: str) -> str:
    """Decode KNIME escape sequences like %%00010 (newline)."""
    if not text:
        return text
    # %%0000X = ASCII code in decimal (5 digits)
    import re
    def replace_escape(m):
        try:
            code = int(m.group(1))
            return chr(code)
        except:
            return m.group(0)
    text = re.sub(r'%%(\d{5})', replace_escape, text)
    return text


def convert_knime_rules_to_python(rules_text: str, output_column: str = "result") -> str:
    """
    P0: Convert KNIME Rule Engine rules to Python np.select code.
    
    KNIME format:
        $Column$ > 10 => "High"
        $Column$ <= 10 => "Low"
        TRUE => "Default"
    
    Python output:
        conditions = [df["Column"] > 10, df["Column"] <= 10]
        choices = ["High", "Low"]
        df["result"] = np.select(conditions, choices, default="Default")
    """
    if not rules_text:
        return f'df["{output_column}"] = "TODO"  # No rules extracted'
    
    lines = rules_text.strip().split('\n')
    conditions = []
    choices = []
    default = '"Unknown"'
    
    for line in lines:
        line = line.strip()
        
        # Skip comments and empty lines
        if not line or line.startswith('//'):
            continue
        
        # Parse: condition => value
        if '=>' not in line:
            continue
            
        parts = line.split('=>', 1)
        if len(parts) != 2:
            continue
        
        condition_str = parts[0].strip()
        value_str = parts[1].strip()
        
        # Handle TRUE/FALSE as default
        if condition_str.upper() in ('TRUE', 'FALSE'):
            default = value_str
            continue
        
        # Convert KNIME column syntax: $Col Name$ -> df["Col Name"]
        condition_py = re.sub(r'\$([^$]+)\$', r'df["\1"]', condition_str)
        
        # Convert operators
        condition_py = condition_py.replace(' AND ', ') & (')
        condition_py = condition_py.replace(' OR ', ') | (')
        
        # Handle NOT at start of condition: NOT expr -> ~(expr)
        if condition_py.startswith('NOT '):
            condition_py = f'~({condition_py[4:]})'
        # Handle NOT in middle of condition
        condition_py = condition_py.replace(' NOT ', ' ~(')
        # Close any opened NOT parentheses before the next operator
        condition_py = re.sub(r'~\(([^)]+)(\s*[&|])', r'~(\1)\2', condition_py)
        
        condition_py = condition_py.replace('=', '==').replace('!==', '!=').replace('<==', '<=').replace('>==', '>=')
        
        # Fix double equals from replacement
        condition_py = re.sub(r'([<>!])={2,}', r'\1=', condition_py)
        
        # Convert KNIME IN operator: df["col"] IN (1,2,3) -> df["col"].isin([1,2,3])
        condition_py = re.sub(r'(df\["[^"]+"\])\s+IN\s*\(([^)]+)\)', r'\1.isin([\2])', condition_py)
        
        # Convert MISSING function: MISSING df["col"] -> df["col"].isna()
        condition_py = re.sub(r'MISSING\s+(df\["[^"]+"\])', r'\1.isna()', condition_py)
        
        # Convert NOT MISSING: NOT MISSING df["col"] -> df["col"].notna()
        condition_py = re.sub(r'NOT\s+MISSING\s+(df\["[^"]+"\])', r'\1.notna()', condition_py)
        
        # Wrap complex conditions
        if ' & ' in condition_py or ' | ' in condition_py:
            condition_py = f'({condition_py})'
        
        conditions.append(condition_py)
        
        # Convert KNIME column syntax in value: $Col Name$ -> df["Col Name"]
        value_py = re.sub(r'\$([^$]+)\$', r'df["\1"]', value_str)
        choices.append(value_py)
    
    if not conditions:
        return f'df["{output_column}"] = {default}  # No valid rules parsed'
    
    # Generate np.select
    cond_str = ', '.join(conditions)
    choice_str = ', '.join(choices)
    
    # Convert KNIME column syntax in default: $Col Name$ -> df["Col Name"]
    default_py = re.sub(r'\$([^$]+)\$', r'df["\1"]', default)
    
    code_lines = [
        f'_conditions = [{cond_str}]',
        f'_choices = [{choice_str}]', 
        f'df["{output_column}"] = np.select(_conditions, _choices, default={default_py})'
    ]
    
    return '\n'.join(code_lines)


def extract_rule_engine_rules(content: str) -> tuple:
    """
    Extract rules from Rule Engine settings.xml using array-based format.
    Returns (rules_text, output_column)
    """
    rules_lines = []
    output_column = "result"
    
    # Extract array-size for rules
    size_match = re.search(r'<config\s+key="rules"[^>]*>.*?<entry\s+key="array-size"\s+type="xint"\s+value="(\d+)"', content, re.DOTALL)
    if size_match:
        array_size = int(size_match.group(1))
        
        # Extract each rule by index
        rules_section = re.search(r'<config\s+key="rules"[^>]*>(.*?)</config>', content, re.DOTALL)
        if rules_section:
            for i in range(array_size):
                rule_match = re.search(rf'<entry\s+key="{i}"\s+type="xstring"\s+value="([^"]*)"', rules_section.group(1))
                if rule_match:
                    rule = decode_knime_escapes(rule_match.group(1))
                    # F1.7: Decode all HTML entities robustly
                    rule = html.unescape(rule)
                    if rule and not rule.startswith('//'):
                        rules_lines.append(rule)
    
    # Fallback: single rules field
    if not rules_lines:
        rules_match = re.search(r'<entry\s+key="rules"\s+[^>]*value="([^"]*)"', content)
        if rules_match:
            rules_text = decode_knime_escapes(rules_match.group(1))
            rules_lines = [l for l in rules_text.split('\n') if l.strip() and not l.strip().startswith('//')]
    
    # Get output column name
    outcol_match = re.search(r'<entry\s+key="new-column-name"\s+[^>]*value="([^"]*)"', content)
    if outcol_match:
        output_column = outcol_match.group(1)
    
    return '\n'.join(rules_lines), output_column


def translate_knime_string_expr(knime_expr: str, target_col: str,
                                 df_var: str = 'df') -> str:
    """Deterministic KNIME StringManipulation → pandas converter.
    
    Handles the most common patterns found in real workflows:
    - toInt(substr($Col$, start, len))
    - toDouble(substr($Col$, start, len))  
    - indexOfChars($Col$, "char")
    - Nested combinations of the above
    
    Returns valid Python assignment or empty string if pattern not recognized.
    """
    import ast as _ast
    
    expr = knime_expr.strip()
    if not expr or not target_col:
        return ''
    
    # Helper: convert $Col$ references to df_var["Col"]
    def col_ref(col_name):
        return f'{df_var}["{col_name}"]'
    
    # Extract all column references
    col_refs = re.findall(r'\$([^$]+)\$', expr)
    
    # Pattern A: toInt(substr($Col$, (indexOfChars($Col$, "char") + N), len))
    m = re.match(
        r'toInt\s*\(\s*substr\s*\(\s*\$([^$]+)\$\s*,\s*'
        r'\(\s*indexOfChars\s*\(\s*\$([^$]+)\$\s*,\s*"([^"]*)"\s*\)\s*\+\s*(\d+)\s*\)\s*,\s*(\d+)\s*\)\s*\)',
        expr
    )
    if m:
        src_col, idx_col, char, offset, length = m.group(1), m.group(2), m.group(3), int(m.group(4)), int(m.group(5))
        # Find position of char, then slice from (pos + offset) for length chars, convert to int
        pos_expr = f'{col_ref(idx_col)}.str.find("{char}")'
        code = (
            f'_pos = {pos_expr}\n'
            f'{col_ref(target_col)} = {col_ref(src_col)}.str.slice(0).values  # placeholder\n'
            # Use apply for row-level dynamic slicing
            f'{col_ref(target_col)} = [{col_ref(src_col)}.iloc[i][p+{offset}:p+{offset}+{length}] '
            f'if p >= 0 else None for i, p in enumerate(_pos)]'
        )
        # Simpler approach: use pandas vectorized string operations
        code = (
            f'_pos = {col_ref(idx_col)}.str.find("{char}")\n'
            f'{col_ref(target_col)} = pd.array(['
            f'{col_ref(src_col)}.iat[i][int(p)+{offset}:int(p)+{offset}+{length}] '
            f'if p >= 0 else None for i, p in enumerate(_pos)], dtype="Int64")'
        )
        # Even simpler: use .apply()
        code = (
            f'{col_ref(target_col)} = {col_ref(src_col)}.apply('
            f'lambda x: int(x[x.find("{char}")+{offset}:x.find("{char}")+{offset}+{length}]) '
            f'if pd.notna(x) and x.find("{char}") >= 0 else None)'
        )
        # Validate
        try:
            _ast.parse(code)
            return code
        except SyntaxError:
            pass
    
    # Pattern B: toInt(substr($Col$, (indexOfChars($Col$, "char") + N), len)) — same as A
    # Already handled above
    
    # Pattern C: toDouble(substr($Col$, 0, indexOfChars($Col$, "char") + N))
    m = re.match(
        r'toDouble\s*\(\s*substr\s*\(\s*\$([^$]+)\$\s*,\s*0\s*,\s*'
        r'indexOfChars\s*\(\s*\$([^$]+)\$\s*,\s*"([^"]*)"\s*\)\s*\+\s*(\d+)\s*\)\s*\)',
        expr
    )
    if m:
        src_col, idx_col, char, offset = m.group(1), m.group(2), m.group(3), int(m.group(4))
        code = (
            f'{col_ref(target_col)} = {col_ref(src_col)}.apply('
            f'lambda x: float(x[:x.find("{char}")+{offset}]) '
            f'if pd.notna(x) and x.find("{char}") >= 0 else None)'
        )
        try:
            _ast.parse(code)
            return code
        except SyntaxError:
            pass
    
    # Pattern D: Simple toInt(substr($Col$, start, len)) — fixed positions
    m = re.match(
        r'toInt\s*\(\s*substr\s*\(\s*\$([^$]+)\$\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*\)',
        expr
    )
    if m:
        src_col, start, length = m.group(1), int(m.group(2)), int(m.group(3))
        code = f'{col_ref(target_col)} = {col_ref(src_col)}.str[{start}:{start + length}].astype(int)'
        try:
            _ast.parse(code)
            return code
        except SyntaxError:
            pass
    
    # Pattern E: Simple toDouble(substr($Col$, start, len))
    m = re.match(
        r'toDouble\s*\(\s*substr\s*\(\s*\$([^$]+)\$\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*\)',
        expr
    )
    if m:
        src_col, start, length = m.group(1), int(m.group(2)), int(m.group(3))
        code = f'{col_ref(target_col)} = pd.to_numeric({col_ref(src_col)}.str[{start}:{start + length}], errors="coerce")'
        try:
            _ast.parse(code)
            return code
        except SyntaxError:
            pass
    
    # Pattern F: substr($Col$, start, len) — just substring, no type conversion
    m = re.match(
        r'substr\s*\(\s*\$([^$]+)\$\s*,\s*(\d+)\s*,\s*(\d+)\s*\)',
        expr
    )
    if m:
        src_col, start, length = m.group(1), int(m.group(2)), int(m.group(3))
        code = f'{col_ref(target_col)} = {col_ref(src_col)}.str[{start}:{start + length}]'
        try:
            _ast.parse(code)
            return code
        except SyntaxError:
            pass
    
    return ''


def translate_java_body(java_code: str, in_cols: list, out_cols: list,
                         df_var: str = 'df') -> str:
    """Deterministic Java Snippet → pandas converter.
    
    Handles simple if/else conditional assignments with:
    - Numeric comparisons: c_X > 0
    - Output assignments: out_Y = value
    - String concatenation: "text" + var.toString() + "text"
    - Multiple output columns
    
    Returns valid Python code or empty string.
    """
    import ast as _ast
    
    if not java_code or not java_code.strip():
        return ''
    
    # Build variable name → column name mapping
    java_to_col = {}
    for knime_name, java_name, _ in in_cols:
        java_to_col[java_name] = knime_name
    out_java_to_col = {}
    for knime_name, java_name, java_type in out_cols:
        out_java_to_col[java_name] = (knime_name, java_type)
    
    # Parse if/else pattern:
    # if (condition) { if_body } else { else_body }
    m = re.search(
        r'if\s*\(\s*(.+?)\s*\)\s*\{(.*?)\}\s*else\s*\{(.*?)\}',
        java_code, re.DOTALL
    )
    if not m:
        # Fallback: simple assignment pattern: out_X = value;
        simple = re.findall(r'(\w+)\s*=\s*(.+?)\s*;', java_code)
        if simple:
            # KNIME system variables → placeholder values
            sys_vars = {'userName': '"system_user"', 'rowCount': '0', 'rowIndex': '0'}
            code_lines = []
            for out_name, val in simple:
                if out_name not in out_java_to_col:
                    # Might be a flow variable — generate comment + placeholder
                    val_py = sys_vars.get(val.strip(), f'"{val.strip()}"')
                    code_lines.append(f'# Flow var: {out_name} = {val_py}')
                    continue
                knime_name, _ = out_java_to_col[out_name]
                val_py = sys_vars.get(val.strip(), f'"{val.strip()}"')
                for jn, kn in java_to_col.items():
                    if jn in val:
                        val_py = f'{df_var}["{kn}"]'
                        break
                code_lines.append(f'{df_var}["{knime_name}"] = {val_py}')
            if code_lines:
                block = '\n'.join(code_lines)
                try:
                    _ast.parse(block)
                    return block
                except SyntaxError:
                    pass
        return ''
    
    cond_raw, if_body, else_body = m.group(1), m.group(2), m.group(3)
    
    # Convert condition: c_VarName > 0 → df["ColName"] > 0
    condition = cond_raw.strip()
    for java_name, knime_name in java_to_col.items():
        condition = condition.replace(java_name, f'{df_var}["{knime_name}"]')
    
    # Extract assignments from if_body and else_body
    def parse_assignments(body):
        assigns = {}
        for stmt in re.findall(r'(\w+)\s*=\s*(.+?)\s*;', body):
            assigns[stmt[0]] = stmt[1]
        return assigns
    
    if_assigns = parse_assignments(if_body)
    else_assigns = parse_assignments(else_body)
    
    # Convert Java values to Python expressions
    def java_val_to_python(val):
        val = val.strip()
        # String concatenation: "text" + var.toString() + "more"
        # Replace .toString() with .astype(str)
        val = re.sub(r'(\w+)\.toString\(\)', r'\1', val)
        # Replace Java variable references with df column refs
        for java_name, knime_name in java_to_col.items():
            val = val.replace(java_name, f'{df_var}["{knime_name}"].astype(str)')
        # Replace output variable refs within concatenation
        for java_name, (knime_name, _) in out_java_to_col.items():
            if java_name in val and '+' in val:
                val = val.replace(java_name, f'{df_var}["{knime_name}"].astype(str)')
        # Replace Java string quotes — already Python compatible
        return val
    
    # Generate np.where() for each output column
    code_lines = []
    all_out_vars = set(if_assigns.keys()) | set(else_assigns.keys())
    
    for out_java_name in all_out_vars:
        if out_java_name not in out_java_to_col:
            continue
        
        knime_name, java_type = out_java_to_col[out_java_name]
        if_val = if_assigns.get(out_java_name, '0')
        else_val = else_assigns.get(out_java_name, '0')
        
        py_if = java_val_to_python(if_val)
        py_else = java_val_to_python(else_val)
        
        line = f'{df_var}["{knime_name}"] = np.where({condition}, {py_if}, {py_else})'
        code_lines.append(line)
    
    if not code_lines:
        return ''
    
    # Validate entire block
    block = '\n'.join(code_lines)
    try:
        _ast.parse(block)
        return block
    except SyntaxError:
        # Try each line individually
        valid_lines = []
        for line in code_lines:
            try:
                _ast.parse(line)
                valid_lines.append(line)
            except SyntaxError:
                continue
        return '\n'.join(valid_lines) if valid_lines else ''


def translate_filter_rules_to_pandas(rules_text: str, df_var: str = 'df') -> str:
    """Convert KNIME Rule Engine Filter rules to a pandas boolean condition.
    
    Input:  '$NuDiffCets$ > 1 OR $NuDiffCets$ < -1 => TRUE'
    Output: '((df["NuDiffCets"] > 1) | (df["NuDiffCets"] < -1))'
    
    Falls back to empty string if rules contain flow variables or
    complex patterns that can't be safely translated.
    """
    if not rules_text or not rules_text.strip():
        return ''
    
    conditions = []
    for line in rules_text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('//'):
            continue
        
        # Skip rules with flow variables ($${VAR}$$) — can't translate statically
        if '$$' in line:
            return ''  # Bail: flow variables present
        
        # Extract the condition part (before =>)
        parts = line.split('=>')
        if len(parts) < 2:
            continue
        cond_part = parts[0].strip()
        result = parts[1].strip().upper()
        
        # Replace $Col$ with df["Col"]
        cond = re.sub(r'\$([^$]+)\$', rf'{df_var}["\1"]', cond_part)
        # Replace KNIME logical operators
        cond = re.sub(r'\bOR\b', '|', cond)
        cond = re.sub(r'\bAND\b', '&', cond)
        cond = re.sub(r'\bNOT\s+', '~', cond)
        # Replace KNIME MISSING check
        cond = re.sub(r'MISSING\s+' + re.escape(df_var) + r'\["([^"]+)"\]',
                      rf'{df_var}["\1"].isna()', cond)
        # Replace = with == for comparison (not <=, >=, !=)
        cond = re.sub(r'(?<![<>!])=(?!=)', '==', cond)
        
        # Safety check: no KNIME-specific patterns remaining
        if re.search(r'[A-Z]{2,}\s', cond) and 'isna' not in cond:
            return ''  # Bail: unrecognized KNIME keyword
        
        cond = f'({cond.strip()})'
        
        if result == 'FALSE':
            cond = f'~{cond}'
        
        conditions.append(cond)
    
    if not conditions:
        return ''
    
    return ' | '.join(conditions)


# P1: KNIME aggregation method to pandas mapping
AGGREGATION_MAP = {
    'Sum': 'sum',
    'Mean': 'mean',
    'Count': 'count',
    'Min': 'min',
    'Max': 'max',
    'Median': 'median',
    'Standard Deviation': 'std',
    'Variance': 'var',
    'First': 'first',
    'Last': 'last',
    'List': lambda x: x.tolist(),
    'Unique count': 'nunique',
    'Mode': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None,
}


def _normalize_agg_method(method: str) -> str:
    """Normalize KNIME aggregation method names (e.g. 'Sum_V2.5.2' -> 'Sum')."""
    # Strip version suffix like _V2.5.2, _V1.0, etc.
    normalized = re.sub(r'_V[\d.]+$', '', method)
    pandas_method = AGGREGATION_MAP.get(normalized, None)
    if pandas_method and isinstance(pandas_method, str):
        return pandas_method
    # Try case-insensitive match
    for k, v in AGGREGATION_MAP.items():
        if k.lower() == normalized.lower() and isinstance(v, str):
            return v
    return 'sum'  # Default fallback


def extract_groupby_config(content: str) -> dict:
    """
    P1: Extract GroupBy configuration from settings.xml.
    Returns dict with group_columns, aggregations list.
    
    KNIME uses:
    - 'grouByColumns' (typo, no 'p') with InclList for group columns
    - 'aggregationColumn' with parallel arrays: columnNames + aggregationMethod
    """
    config = {'group_columns': [], 'aggregations': {}}
    
    # 1. Extract group columns - KNIME uses "grouByColumns" (typo without 'p')
    for key_name in ['grouByColumns', 'groupByColumns', 'columnKeys']:
        grp_section = re.search(
            rf'<config\s+key="{key_name}"[^>]*>(.*?)</config>',
            content, re.DOTALL
        )
        if grp_section:
            # Look for InclList sub-section first
            incl_match = re.search(
                r'<config\s+key="InclList"[^>]*>(.*?)</config>',
                grp_section.group(1), re.DOTALL
            )
            search_in = incl_match.group(1) if incl_match else grp_section.group(1)
            cols = re.findall(
                r'<entry\s+key="\d+"\s+type="xstring"\s+value="([^"]+)"',
                search_in
            )
            if cols:
                config['group_columns'] = cols
                break
    
    # 2. Extract aggregations - search columnNames and aggregationMethod
    #    directly in content (nested XML makes regex sectioning unreliable)
    names_section = re.search(
        r'<config\s+key="columnNames"[^>]*>(.*?)</config>',
        content, re.DOTALL
    )
    methods_section = re.search(
        r'<config\s+key="aggregationMethod"[^>]*>(.*?)</config>',
        content, re.DOTALL
    )
    
    if names_section and methods_section:
        col_names = re.findall(
            r'<entry\s+key="\d+"\s+type="xstring"\s+value="([^"]+)"',
            names_section.group(1)
        )
        methods = re.findall(
            r'<entry\s+key="\d+"\s+type="xstring"\s+value="([^"]+)"',
            methods_section.group(1)
        )
        for col, method in zip(col_names, methods):
            config['aggregations'][col] = _normalize_agg_method(method)
    
    return config


def generate_groupby_code(config: dict) -> str:
    """Generate pandas groupby().agg() code from extracted config."""
    if not config.get('group_columns'):
        return 'df = df.copy()  # GroupBy: no group columns extracted'
    
    group_cols = config['group_columns']
    aggregations = config.get('aggregations', {})
    
    group_str = ', '.join(f'"{c}"' for c in group_cols)
    
    if aggregations:
        agg_items = ', '.join(f'"{col}": "{method}"' for col, method in aggregations.items())
        return f'df = df.groupby([{group_str}]).agg({{{agg_items}}}).reset_index()'
    else:
        return f'df = df.groupby([{group_str}]).first().reset_index()  # No aggregations specified'


def extract_knwf(knwf_path, log):
    """Extract .knwf to temp directory."""
    try:
        temp_dir = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(knwf_path, 'r') as zf:
            zf.extractall(temp_dir)
        return temp_dir
    except Exception as e:
        log.add_error(f"Extraction failed: {e}")
        raise


# ── Generic DB Info Extraction ──────────────────────────────
# Works for ANY database node type found in KNIME workflows.
# Extracts maximum information for AI-assisted connection build.

# Keys we look for in ANY DB node settings.xml
_DB_KEYS = [
    'driver', 'database', 'database_name', 'db_type', 'db_dialect',
    'db_driver', 'host', 'port', 'schema', 'table', 'catalog',
    'user', 'userName', 'credential', 'auth_type',
    'service_account_email', 'key_file_location',
    'project', 'project_id', 'dataset', 'location', 'region',
    'warehouse', 'role', 'account', 'instance',
    'timeout', 'connection_timeout', 'query_timeout',
    'url', 'server', 'server_name', 'sid', 'service_name',
    'statement', 'sql_statement', 'query',
    'insert_mode', 'if_exists', 'batch_size',
    'port_object_summary',
]

# Factory-name fragments → normalized db_type
_DB_TYPE_MAP = {
    'BigQuery': 'bigquery',
    'GoogleApi': 'google_api',
    'JDBC': 'jdbc',
    'MySQL': 'mysql',
    'Postgres': 'postgres',
    'SQLServer': 'mssql',
    'MSSQL': 'mssql',
    'Oracle': 'oracle',
    'Hive': 'hive',
    'Snowflake': 'snowflake',
    'Redshift': 'redshift',
    'SQLite': 'sqlite',
    'Mongo': 'mongodb',
    'Cassandra': 'cassandra',
    'H2': 'h2',
    'DB2': 'db2',
    'Sybase': 'sybase',
    'Teradata': 'teradata',
    'Vertica': 'vertica',
    'MariaDB': 'mariadb',
    'DBLoader': 'db_loader',
    'DBReader': 'db_reader',
    'DBWriter': 'db_writer',
    'DBQueryReader': 'db_query_reader',
    'DatabaseLooping': 'db_looping',
    'DatabaseConnector': 'jdbc',
}

# Factory fragments that indicate a DB-related node
_DB_FACTORY_HINTS = list(_DB_TYPE_MAP.keys()) + [
    'Connector', 'Database',
]


def _extract_db_info(content: str, factory: str) -> dict:
    """Extract maximum DB connection info from a KNIME node settings.xml.

    Returns a dict with all discovered keys, or empty dict if not a DB node.
    The returned dict always contains 'db_type' if any DB info was found.
    """
    short = factory.split('.')[-1]

    # ── 0. Is this a DB-related node? ────────────────────────
    if not any(hint in short for hint in _DB_FACTORY_HINTS):
        return {}

    info = {}
    
    # Mark actual connector/authentication nodes (for header generation)
    if 'Connector' in short or 'Api' in short:
        info['is_connector'] = True

    # ── 1. Determine db_type from factory name ───────────────
    for fragment, dtype in _DB_TYPE_MAP.items():
        if fragment in short:
            info['db_type'] = dtype
            break
    if 'db_type' not in info:
        info['db_type'] = 'unknown'

    # ── 2. Extract all known keys ────────────────────────────
    for key in _DB_KEYS:
        m = re.search(
            rf'<entry\s+key="{re.escape(key)}"\s+[^>]*value="([^"]*)"',
            content,
        )
        if m and m.group(1):
            val = m.group(1).strip()
            # Skip KNIME placeholder values
            if val.startswith('<') and val.endswith('>'):
                continue
            if val.startswith('[Workflows edited'):
                continue
            # Skip KNIME internal classes
            if 'org.knime.' in val and key not in ('port_object_summary',):
                continue
            info[key] = val

    # Integer keys (port, timeout)
    for key in ['port', 'timeout', 'connection_timeout', 'query_timeout',
                'batch_size']:
        m = re.search(
            rf'<entry\s+key="{re.escape(key)}"\s+type="xint"\s+value="(\d+)"',
            content,
        )
        if m:
            info[key] = int(m.group(1))

    # ── 3. Extract JDBC URLs anywhere in content ─────────────
    jdbc_urls = re.findall(r'(jdbc:[a-zA-Z0-9+:/_.\-@]+)', content)
    if jdbc_urls:
        info['jdbc_url'] = jdbc_urls[0]
        # Parse JDBC URL components
        _parse_jdbc_url(info, jdbc_urls[0])

    # ── 4. Detect driver from content (not just "driver" key) ─
    drv_m = re.search(
        r'<entry\s+key="driver"\s+[^>]*value="([^"]*)"', content
    )
    if drv_m and 'org.knime' not in drv_m.group(1):
        info['jdbc_driver'] = drv_m.group(1)
        # Infer db_type from driver class
        if info.get('db_type') in ('unknown', 'jdbc', 'db_looping',
                                    'db_reader'):
            info['db_type'] = _infer_type_from_driver(drv_m.group(1))

    # ── 5. Extract BQ project from SQL (fallback) ────────────
    for sql_key in ('sql_statement', 'statement', 'query'):
        if sql_key in info:
            sql = info[sql_key]
            bq_m = re.search(
                r'`([a-zA-Z0-9_-]+)\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+`',
                sql,
            )
            if bq_m:
                info['bq_project_from_sql'] = bq_m.group(1)

    # ── 6. Extract SQL comment hints (INSTANCIA, BANCO) ──────
    for sql_key in ('sql_statement', 'statement', 'query'):
        if sql_key in info:
            sql_raw = info[sql_key]
            inst_m = re.search(r'INSTANCIA:\s*(\S+)', sql_raw)
            db_m = re.search(r'BANCO DE DADOS:\s*([^\n*]+)', sql_raw)
            if inst_m:
                info['sql_hint_instance'] = inst_m.group(1)
            if db_m:
                info['sql_hint_database'] = db_m.group(1).strip()

    # ── 7. Normalize fields for header generator ─────────────
    # bq_project: prefer database_name, then database, then from SQL
    if info.get('db_type') == 'bigquery':
        for k in ('database_name', 'database', 'bq_project_from_sql'):
            if k in info and info[k]:
                info['bq_project'] = info[k]
                break

    # service_account: propagate from key
    if 'service_account_email' in info:
        info['service_account'] = info['service_account_email']

    # key_file: propagate
    if 'key_file_location' in info:
        info['key_file_path'] = info['key_file_location']

    # DBLoader: mark action
    if 'Loader' in short or 'Writer' in short:
        info['db_action'] = 'load'
        if 'table' in info:
            info['db_table'] = info['table']
        if 'schema' in info:
            info['db_schema'] = info['schema']
        if 'database' in info and info.get('db_type') == 'db_loader':
            info['bq_project'] = info['database']

    return info


def _parse_jdbc_url(info: dict, url: str) -> None:
    """Parse a JDBC URL into host, port, database components."""
    # jdbc:subprotocol:subname
    # Common patterns:
    #   jdbc:mysql://host:port/db
    #   jdbc:postgresql://host:port/db
    #   jdbc:oracle:thin:@host:port:sid
    #   jdbc:sybase:Tds:host:port
    #   jdbc:hive2://host:port/db
    #   jdbc:sqlserver://host:port;databaseName=db
    parts = url.split(':')
    if len(parts) >= 3:
        info['jdbc_subprotocol'] = parts[1]
    # Try //host:port/db pattern
    m = re.search(r'//([^:/]+):?(\d+)?/?([^;?]*)', url)
    if m:
        if m.group(1):
            info.setdefault('host', m.group(1))
        if m.group(2):
            info.setdefault('port', int(m.group(2)))
        if m.group(3):
            info.setdefault('database_name', m.group(3))
    # Try host:port pattern (Sybase, etc.)
    elif len(parts) >= 5:
        info.setdefault('host', parts[3])
        try:
            info.setdefault('port', int(parts[4].split('/')[0]))
        except (ValueError, IndexError):
            pass
    # sqlserver databaseName=x
    db_m = re.search(r'databaseName=([^;]+)', url)
    if db_m:
        info.setdefault('database_name', db_m.group(1))


def _infer_type_from_driver(driver: str) -> str:
    """Infer database type from JDBC driver class name."""
    driver_lower = driver.lower()
    _DRIVER_MAP = {
        'sybase': 'sybase', 'mysql': 'mysql', 'postgres': 'postgres',
        'oracle': 'oracle', 'sqlserver': 'mssql', 'jtds': 'mssql',
        'hive': 'hive', 'snowflake': 'snowflake', 'redshift': 'redshift',
        'sqlite': 'sqlite', 'h2': 'h2', 'db2': 'db2',
        'teradata': 'teradata', 'vertica': 'vertica',
        'mariadb': 'mariadb', 'bigquery': 'bigquery',
    }
    for fragment, dtype in _DRIVER_MAP.items():
        if fragment in driver_lower:
            return dtype
    return 'jdbc'


def find_nodes_from_settings(extract_dir, log):
    """Find nodes by scanning settings.xml files and extract SQL queries."""
    nodes = []
    node_id = 0
    
    # Patterns indicating a DB node (by content, not just factory)
    db_content_patterns = ['DBReaderNodeFactory', 'DBQueryReaderNodeFactory', 'DatabaseLoopingNodeFactory',
                           'DBLoaderNodeFactory', 'DatabaseConnectorNodeFactory',
                           'GoogleApiConnectorFactory', 'BigQueryDBConnectorNodeFactory']
    
    for settings_file in extract_dir.rglob('settings.xml'):
        try:
            content = settings_file.read_text(encoding='utf-8', errors='ignore')
            
            # Try multiple factory patterns (org.knime, com.knime, legacy nodes)
            factory_match = re.search(r'(?:org|com)\.[a-zA-Z0-9_.]+NodeFactory[a-zA-Z0-9_$]*', content)
            # Fallback: also match *Factory (e.g. GoogleApiConnectorFactory)
            if not factory_match:
                factory_match = re.search(r'(?:org|com)\.[a-zA-Z0-9_.]+(?:Connector|Api)Factory[a-zA-Z0-9_$]*', content)
            
            if factory_match:
                factory = factory_match.group(0)
            else:
                # Fallback: check if it's a DB node by content patterns
                is_db_node = any(p in content for p in db_content_patterns)
                if is_db_node:
                    # Extract factory name from file content
                    for p in db_content_patterns:
                        if p in content:
                            factory = p
                            break
                else:
                    continue  # Not a recognizable node
            
            name = settings_file.parent.name
            name_clean = re.sub(r'\s*\(#\d+\)$', '', name)
            
            node_info = {
                'id': node_id,
                'name': name_clean,
                'factory': factory,
                'path': str(settings_file),
                'sql_query': '',  # Will be populated for DB nodes
            }
            
            # Extract SQL for database nodes - check by content patterns OR factory name
            is_sql_node = any(p in content for p in ['DBReader', 'DBQueryReader', 'DatabaseLooping'])
            is_sql_node = is_sql_node or any(p in factory for p in ['DBReader', 'DBQueryReader', 'DatabaseLooping'])
            
            if is_sql_node:
                sql = extract_sql_from_settings(content)
                if sql:
                    node_info['sql_query'] = sql
                    log.add_warning(f"SQL extracted from {name_clean}")
            
            nodes.append(node_info)
            node_id += 1
                
        except Exception as e:
            log.add_warning(f"Error reading {settings_file.name}: {e}")
    
    return nodes


def extract_sql_from_settings(content: str) -> str:
    """Extract SQL query from settings.xml content."""
    sql_query = ""
    
    # Pattern 1: key="statement" value="..."
    match = re.search(r'<entry\s+key="statement"\s+[^>]*value="([^"]*)"', content, re.IGNORECASE)
    if match:
        sql_query = match.group(1)
    
    # Pattern 2: key="query" value="..."
    if not sql_query:
        match = re.search(r'<entry\s+key="query"\s+[^>]*value="([^"]*)"', content, re.IGNORECASE)
        if match:
            sql_query = match.group(1)
    
    # Pattern 3: value attribute containing SELECT
    if not sql_query:
        matches = re.findall(r'value="([^"]*SELECT[^"]*)"', content, re.IGNORECASE)
        if matches:
            sql_query = max(matches, key=len)
    
    # Decode KNIME escape sequences
    sql_query = decode_knime_escapes(sql_query)
    
    # Decode HTML entities
    sql_query = sql_query.replace('&lt;', '<')
    sql_query = sql_query.replace('&gt;', '>')
    sql_query = sql_query.replace('&amp;', '&')
    sql_query = sql_query.replace('&quot;', '"')
    sql_query = sql_query.replace('&#xA;', '\n')
    sql_query = sql_query.replace('&#xD;', '\r')
    sql_query = sql_query.replace('&#x9;', '\t')
    
    return sql_query.strip()



def enrich_nodes_with_params(nodes, extract_dir, log):
    """Enrich nodes with parameters extracted from settings.xml files."""
    # Build maps for different node types
    params_by_name = {}
    params_by_rel_path = {}  # Map by relative path for precise JSON matching
    
    for settings_file in extract_dir.rglob('settings.xml'):
        try:
            content = settings_file.read_text(encoding='utf-8', errors='ignore')
            name = settings_file.parent.name
            name_clean = re.sub(r'\s*\(#\d+\)$', '', name)
            
            # Compute relative path for JSON settings_path matching
            try:
                rel_path = str(settings_file.relative_to(extract_dir)).replace('\\', '/')
            except ValueError:
                rel_path = None
            
            node_params = {}
            
            # SQL nodes (Database Reader, Looping, Query Reader)
            if any(p in content for p in ['DBReader', 'DBQueryReader', 'DatabaseLooping']):
                sql = extract_sql_from_settings(content)
                if sql:
                    node_params['sql_query'] = sql
                    log.add_warning(f"SQL extracted: {name_clean}")
            
            # ── Generic DB info extraction ──────────────────────────
            #   Works for ANY database node: JDBC, BigQuery, MySQL,
            #   Oracle, Postgres, Hive, Snowflake, etc.
            factory_m = re.search(r'(?:org|com)\.[a-zA-Z0-9_.]+NodeFactory[a-zA-Z0-9_$]*', content)
            if not factory_m:
                factory_m = re.search(r'(?:org|com)\.[a-zA-Z0-9_.]+(?:Connector|Api)Factory[a-zA-Z0-9_$]*', content)
            local_factory = factory_m.group(0) if factory_m else ''
            db_info = _extract_db_info(content, local_factory)
            if db_info:
                node_params.update(db_info)
            
            # Math Formula (JEP) - expression
            if 'JEPNodeFactory' in content:
                match = re.search(r'<entry\s+key="expression"\s+[^>]*value="([^"]*)"', content)
                if match:
                    expr = decode_knime_escapes(match.group(1))
                    expr = html.unescape(expr)  # F1.7: decode &gt; &lt; etc.
                    # Convert KNIME column syntax $Col$ to Python df["Col"]
                    python_expr = re.sub(r'\$([^$]+)\$', r'df["\1"]', expr)
                    # F1.5: JEP uses ^ for power, Python uses **
                    python_expr = re.sub(r'(?<!\w)\^(?!\w)', '**', python_expr)
                    # JEP ternary: if(cond, a, b) -> np.where(cond, a, b)
                    python_expr = re.sub(r'\bif\s*\(', 'np.where(', python_expr)
                    node_params['expression'] = python_expr
                    node_params['original_expression'] = expr
                # Target column
                col_match = re.search(r'<entry\s+key="replaced_column"\s+[^>]*value="([^"]*)"', content)
                if col_match and col_match.group(1):
                    node_params['target_column'] = col_match.group(1)
                else:
                    # New column
                    new_col = re.search(r'<entry\s+key="new_column_name"\s+[^>]*value="([^"]*)"', content)
                    if new_col:
                        node_params['target_column'] = new_col.group(1)
            
            # Rule Engine - rules and output column (P0: Enhanced extraction)
            if 'RuleEngineNodeFactory' in content:
                rules_text, output_col = extract_rule_engine_rules(content)
                if rules_text:
                    node_params['rules'] = rules_text
                    node_params['output_column'] = output_col
                    node_params['python_rules'] = convert_knime_rules_to_python(rules_text, output_col)
            
            # GroupBy - columns and aggregations (P1: Enhanced extraction)
            if 'GroupByNodeFactory' in content:
                groupby_config = extract_groupby_config(content)
                if groupby_config['group_columns']:
                    node_params['group_columns'] = groupby_config['group_columns']
                    node_params['aggregations'] = groupby_config.get('aggregations', {})
                    node_params['groupby_code'] = generate_groupby_code(groupby_config)
            
            # Column Filter - included columns (F1.2: improved InclList extraction)
            if 'DataColumnSpecFilterNodeFactory' in content or 'ColumnFilterNodeFactory' in content:
                # Try nested InclList first (most reliable)
                incl_match = re.search(
                    r'<config\s+key="column-filter"[^>]*>.*?'
                    r'<config\s+key="included_names"[^>]*>(.*?)</config>',
                    content, re.DOTALL
                )
                if incl_match:
                    included = re.findall(
                        r'<entry\s+key="\d+"\s+type="xstring"\s+value="([^"]+)"',
                        incl_match.group(1)
                    )
                    if included:
                        node_params['included_columns'] = included
                else:
                    # Fallback: direct column-filter section
                    filter_section = re.search(r'<config\s+key="column-filter">(.*?)</config>', content, re.DOTALL)
                    if filter_section:
                        included = re.findall(r'<entry\s+key="\d+"\s+type="xstring"\s+value="([^"]+)"', filter_section.group(1))
                        if included:
                            node_params['included_columns'] = included
            
            # Column Rename - mappings
            if 'RenameNodeFactory' in content:
                old_names = re.findall(r'<entry\s+key="old_column_name"\s+[^>]*value="([^"]*)"', content)
                new_names = re.findall(r'<entry\s+key="new_column_name"\s+[^>]*value="([^"]*)"', content)
                if old_names and new_names:
                    node_params['rename_map'] = dict(zip(old_names, new_names))
            
            # Row Filter - extract filter type and column (F1.5)
            if 'RowFilterNodeFactory' in content or 'RowFilter2PortNodeFactory' in content:
                type_match = re.search(r'<entry\s+key="RowFilter_TypeID"\s+[^>]*value="([^"]*)"', content)
                col_match = re.search(r'<entry\s+key="(?:column_name|ColumnName)"\s+[^>]*value="([^"]*)"', content)
                include_match = re.search(r'<entry\s+key="include"\s+[^>]*value="([^"]*)"', content)
                if type_match:
                    filter_type = type_match.group(1)
                    column = col_match.group(1) if col_match else ''
                    is_include = include_match.group(1) == 'true' if include_match else True
                    node_params['filter_type'] = filter_type
                    node_params['filter_column'] = column
                    node_params['filter_include'] = is_include
                    # Generate pandas filter code based on TypeID
                    if filter_type == 'StringComp_RowFilter':
                        pattern = re.search(r'<entry\s+key="StringValue"\s+[^>]*value="([^"]*)"', content)
                        val = pattern.group(1) if pattern else ''
                        if is_include:
                            node_params['filter_code'] = f'df = df[df["{column}"] == "{val}"]'
                        else:
                            node_params['filter_code'] = f'df = df[df["{column}"] != "{val}"]'
                    elif filter_type == 'RangeVal_RowFilter':
                        lower = re.search(r'<entry\s+key="LowerBound"\s+[^>]*value="([^"]*)"', content)
                        upper = re.search(r'<entry\s+key="UpperBound"\s+[^>]*value="([^"]*)"', content)
                        # Also try DoubleCell/IntCell for range values
                        if not lower:
                            lower = re.search(r'<entry\s+key="(?:DoubleCell|IntCell)"\s+[^>]*value="([^"]*)"', content)
                        lower_v = lower.group(1) if lower else None
                        upper_v = upper.group(1) if upper else None
                        conditions = []
                        if lower_v:
                            conditions.append(f'df["{column}"] >= {lower_v}')
                        if upper_v:
                            conditions.append(f'df["{column}"] <= {upper_v}')
                        if conditions:
                            cond = ' & '.join(f'({c})' for c in conditions)
                            if is_include:
                                node_params['filter_code'] = f'df = df[{cond}]'
                            else:
                                node_params['filter_code'] = f'df = df[~({cond})]'
                    elif filter_type == 'MissingVal_RowFilter':
                        if is_include:
                            node_params['filter_code'] = f'df = df[df["{column}"].isna()]'
                        else:
                            node_params['filter_code'] = f'df = df[df["{column}"].notna()]'
                    elif filter_type == 'ColVal_RowFilter':
                        node_params['filter_code'] = f'df = df[df["{column}"].notna()]  # ColVal filter'
            
            # Sorter - extract sort columns and direction (F1.4)
            if 'SorterNodeFactory' in content:
                sort_section = re.search(r'<config\s+key="incllist"[^>]*>(.*?)</config>', content, re.DOTALL)
                if sort_section:
                    sort_cols = re.findall(r'<entry\s+key="\d+"\s+type="xstring"\s+value="([^"]+)"', sort_section.group(1))
                    if sort_cols:
                        node_params['sort_columns'] = sort_cols
                order_section = re.search(r'<config\s+key="sortOrder"[^>]*>(.*?)</config>', content, re.DOTALL)
                if order_section:
                    orders = re.findall(r'<entry\s+key="\d+"\s+type="xboolean"\s+value="([^"]+)"', order_section.group(1))
                    node_params['sort_ascending'] = [o == 'true' for o in orders]
            
            # F3.1: String Manipulation - extract expression and convert functions
            if 'StringManipulationNodeFactory' in content:
                match = re.search(r'<entry\s+key="expression"\s+[^>]*value="([^"]*)"', content)
                if match:
                    expr = decode_knime_escapes(match.group(1))
                    expr = html.unescape(expr)
                    # Convert KNIME $Col$ → df["Col"]
                    py_expr = re.sub(r'\$([^$]+)\$', r'df["\1"]', expr)
                    # KNIME string functions → Python
                    py_expr = py_expr.replace('join(', '"+".join([')  # approximate
                    py_expr = re.sub(r'\bupperCase\s*\(', '.upper() #(', py_expr)
                    py_expr = re.sub(r'\blowerCase\s*\(', '.lower() #(', py_expr)
                    py_expr = re.sub(r'\blength\s*\(', 'len(', py_expr)
                    py_expr = re.sub(r'\bsubstr\s*\(', '[', py_expr)  # needs manual
                    py_expr = re.sub(r'\breplace\s*\(', '.str.replace(', py_expr)
                    py_expr = re.sub(r'\btoInt\s*\(', 'int(', py_expr)
                    py_expr = re.sub(r'\btoDouble\s*\(', 'float(', py_expr)
                    py_expr = re.sub(r'\btoString\s*\(', 'str(', py_expr)
                    py_expr = re.sub(r'\bstrip\s*\(', '.str.strip(', py_expr)
                    py_expr = re.sub(r'\btrim\s*\(', '.str.strip(', py_expr)
                    py_expr = re.sub(r'\bindexOf\s*\(', '.str.find(', py_expr)
                    py_expr = re.sub(r'\bstring\s*\(', 'str(', py_expr)
                    node_params['string_expr'] = py_expr
                    node_params['original_expression'] = expr
                col_match = re.search(r'<entry\s+key="replaced_column"\s+[^>]*value="([^"]*)"', content)
                if col_match and col_match.group(1):
                    node_params['target_column'] = col_match.group(1)
            
            # F3.2: Rule Engine Filter - extract rules for row filtering
            if 'RuleEngineFilterNodeFactory' in content:
                rules, _ = extract_rule_engine_rules(content)
                if rules:
                    node_params['filter_rules'] = rules
                    node_params['rules_mode'] = 'filter'
            
            # F6: Java Snippet — extract scriptBody + column mappings
            if 'JavaSnippetNodeFactory' in content or 'JavaEditVarNodeFactory' in content:
                body_match = re.search(r'key="scriptBody"\s+[^>]*value="([^"]*)"', content)
                if body_match:
                    java_raw = html.unescape(body_match.group(1))
                    java_raw = java_raw.replace('%%00010', '\n').replace('%%00009', '\t')
                    java_raw = java_raw.replace('%00010', '\n').replace('%00009', '\t')
                    # Remove comment-only lines
                    java_lines = [l for l in java_raw.split('\n')
                                  if l.strip() and not l.strip().startswith('//')]
                    node_params['java_body'] = '\n'.join(java_lines)
                    
                    # Extract column mappings using section markers
                    # (nested <config> breaks simple (.*?)</config> regex)
                    java_in_cols = []
                    java_out_cols = []
                    
                    # Find section boundaries
                    in_start = content.find('"inCols"')
                    out_start = content.find('"outCols"')
                    outv_start = content.find('"outVars"')
                    
                    # Extract inCols section (between inCols and either outCols or end)
                    if in_start >= 0:
                        in_end = out_start if out_start > in_start else outv_start if outv_start > in_start else len(content)
                        in_section = content[in_start:in_end]
                        # Find all Name/JavaName/JavaType triplets
                        names = re.findall(r'key="Name"[^>]*value="([^"]*)"', in_section)
                        jnames = re.findall(r'key="JavaName"[^>]*value="([^"]*)"', in_section)
                        jtypes = re.findall(r'key="JavaType"[^>]*value="([^"]*)"', in_section)
                        for idx in range(min(len(names), len(jnames))):
                            jt = jtypes[idx] if idx < len(jtypes) else ''
                            java_in_cols.append((names[idx], jnames[idx], jt))
                    
                    # Extract outCols section (between outCols and outVars)
                    if out_start >= 0:
                        out_end = outv_start if outv_start > out_start else len(content)
                        out_section = content[out_start:out_end]
                        names = re.findall(r'key="Name"[^>]*value="([^"]*)"', out_section)
                        jnames = re.findall(r'key="JavaName"[^>]*value="([^"]*)"', out_section)
                        jtypes = re.findall(r'key="JavaType"[^>]*value="([^"]*)"', out_section)
                        for idx in range(min(len(names), len(jnames))):
                            jt = jtypes[idx] if idx < len(jtypes) else ''
                            java_out_cols.append((names[idx], jnames[idx], jt))
                    
                    node_params['java_in_cols'] = java_in_cols
                    node_params['java_out_cols'] = java_out_cols
            
            # F7: Python2Script — extract sourceCode
            if 'Python2ScriptNodeFactory' in content or 'PythonScriptNodeFactory' in content:
                src_match = re.search(r'key="sourceCode"\s+[^>]*value="([^"]*)"', content)
                if src_match:
                    py_raw = html.unescape(src_match.group(1))
                    py_raw = py_raw.replace('%%00010', '\n').replace('%%00009', '\t')
                    py_raw = py_raw.replace('%00010', '\n').replace('%00009', '\t')
                    py_raw = py_raw.replace('%%00013', '').replace('%00013', '')
                    # Clean: remove empty leading lines
                    py_lines = py_raw.split('\n')
                    while py_lines and not py_lines[0].strip():
                        py_lines.pop(0)
                    while py_lines and not py_lines[-1].strip():
                        py_lines.pop()
                    node_params['python_source'] = '\n'.join(py_lines)
            
            # F3.3: Constant Value Column
            if 'ConstantValueColumnNodeFactory' in content:
                val_match = re.search(r'<entry\s+key="column-value"\s+[^>]*value="([^"]*)"', content)
                if not val_match:
                    val_match = re.search(r'<entry\s+key="value"\s+[^>]*value="([^"]*)"', content)
                col_match = re.search(r'<entry\s+key="new-column-name"\s+[^>]*value="([^"]*)"', content)
                if not col_match:
                    col_match = re.search(r'<entry\s+key="new_column_name"\s+[^>]*value="([^"]*)"', content)
                type_match = re.search(r'<entry\s+key="column-type"\s+[^>]*value="([^"]*)"', content)
                if not type_match:
                    type_match = re.search(r'<entry\s+key="selected_new_column_type"\s+[^>]*value="([^"]*)"', content)
                if val_match:
                    node_params['const_value'] = val_match.group(1)
                if col_match:
                    node_params['const_column'] = col_match.group(1)
                if type_match:
                    node_params['const_type'] = type_match.group(1)
            
            # F3.4: Date/Time nodes
            if 'OldToNewTimeNodeFactory' in content:
                node_params['date_action'] = 'old_to_new'
                cols = _extract_column_list(content)
                if cols:
                    node_params['date_columns'] = cols
                else:
                    col_match = re.search(r'<entry\s+key="selectedColumn"\s+[^>]*value="([^"]*)"', content)
                    if col_match:
                        node_params['date_column'] = col_match.group(1)
            
            if 'NewToOldTimeNodeFactory' in content:
                node_params['date_action'] = 'new_to_old'
                cols = _extract_column_list(content)
                if cols:
                    node_params['date_columns'] = cols
                else:
                    col_match = re.search(r'<entry\s+key="selectedColumn"\s+[^>]*value="([^"]*)"', content)
                    if col_match:
                        node_params['date_column'] = col_match.group(1)
            
            if 'DateTimeDifferenceNodeFactory' in content:
                node_params['date_action'] = 'difference'
                for k in ['col1', 'col2', 'first_column', 'second_column',
                           'col_select1', 'col_select2']:
                    m = re.search(rf'<entry\s+key="{k}"\s+[^>]*value="([^"]*)"', content)
                    if m:
                        # Normalize to col1/col2
                        dest = 'col1' if '1' in k else 'col2'
                        node_params[dest] = m.group(1)
                gran_match = re.search(r'<entry\s+key="granularity"\s+[^>]*value="([^"]*)"', content)
                if gran_match:
                    node_params['granularity'] = gran_match.group(1)
                out_match = re.search(r'<entry\s+key="(?:output_column_name|new_col_name)"\s+[^>]*value="([^"]*)"', content)
                if out_match:
                    node_params['output_column'] = out_match.group(1)
            
            if 'CreateDateTimeNodeFactory' in content:
                node_params['date_action'] = 'create'
            if 'ExtractDateTimeFieldsNodeFactory' in content:
                node_params['date_action'] = 'extract_fields'
            if 'DateTimeShiftNodeFactory' in content:
                node_params['date_action'] = 'shift'
            if 'ModifyTimeNodeFactory' in content:
                node_params['date_action'] = 'modify'
            
            # F3.5: RoundDouble - extract decimal places
            if 'RoundDoubleNodeFactory' in content:
                prec_match = re.search(r'<entry\s+key=".*?precision.*?"\s+[^>]*value="(\d+)"', content, re.I)
                if prec_match:
                    node_params['precision'] = int(prec_match.group(1))
                col_match = re.search(r'<entry\s+key="column_name"\s+[^>]*value="([^"]*)"', content)
                if col_match:
                    node_params['round_column'] = col_match.group(1)
            
            # F3.6: TableToVariable
            if 'TableToVariable3NodeFactory' in content or 'TableToVariableNode' in content:
                node_params['action'] = 'table_to_variable'
            
            # F3.7: NumberToString / StringToNumber / DoubleToInt
            if 'NumberToStringNodeFactory' in content or 'NumberToString2NodeFactory' in content:
                node_params['action'] = 'number_to_string'
                cols = _extract_column_list(content)
                if cols:
                    node_params['convert_columns'] = cols
            
            if 'StringToNumber2NodeFactory' in content or 'StringToNumberNodeFactory' in content:
                node_params['action'] = 'string_to_number'
                cols = _extract_column_list(content)
                if cols:
                    node_params['convert_columns'] = cols
            
            if 'DoubleToIntNodeFactory' in content:
                node_params['action'] = 'double_to_int'
                cols = _extract_column_list(content)
                if cols:
                    node_params['convert_columns'] = cols
            
            # F3.8: VariableToTable
            if 'VariableToTable4NodeFactory' in content or 'VariableToTableNode' in content:
                node_params['action'] = 'variable_to_table'
            
            # F3.9: ColumnResorter
            if 'ColumnResorterNodeFactory' in content:
                node_params['action'] = 'column_resorter'
                # Try named config sections first, then model entries
                cols = None
                for cfg_key in ['newOrder', 'column-order']:
                    order_section = re.search(rf'<config\s+key="{cfg_key}"[^>]*>(.*?)</config>', content, re.DOTALL)
                    if order_section:
                        cols = re.findall(r'<entry\s+key="\d+"\s+type="xstring"\s+value="([^"]+)"', order_section.group(1))
                        if cols:
                            break
                if not cols:
                    # Fallback: numbered entries in model section
                    model_section = re.search(r'<config\s+key="model"[^>]*>(.*?)</config>', content, re.DOTALL)
                    if model_section:
                        cols = re.findall(r'<entry\s+key="\d+"\s+type="xstring"\s+value="([^"]+)"', model_section.group(1))
                if cols:
                    node_params['resorter_columns'] = cols
            
            # F3.10: RowFilter2Port — extract condition
            if 'RowFilter2PortNodeFactory' in content:
                node_params['action'] = 'row_splitter'
                # Try DataValueFilter pattern
                filter_col = re.search(r'<entry\s+key="column"\s+[^>]*value="([^"]*)"', content)
                filter_op = re.search(r'<entry\s+key="operator"\s+[^>]*value="([^"]*)"', content)
                filter_val = re.search(r'<entry\s+key="value"\s+[^>]*value="([^"]*)"', content)
                if filter_col:
                    node_params['filter_column'] = filter_col.group(1)
                if filter_op:
                    node_params['filter_operator'] = filter_op.group(1)
                if filter_val:
                    node_params['filter_value'] = filter_val.group(1)
            
            # Joiner - extract join columns and type (F1.1)
            if 'Joiner3NodeFactory' in content or 'JoinerNodeFactory' in content:
                left_section = re.search(r'<config\s+key="leftTableJoinPredicate"[^>]*>(.*?)</config>', content, re.DOTALL)
                right_section = re.search(r'<config\s+key="rightTableJoinPredicate"[^>]*>(.*?)</config>', content, re.DOTALL)
                if left_section:
                    left_cols = re.findall(r'<entry\s+key="\d+"\s+type="xstring"\s+value="([^"]+)"', left_section.group(1))
                    node_params['left_join_cols'] = left_cols
                if right_section:
                    right_cols = re.findall(r'<entry\s+key="\d+"\s+type="xstring"\s+value="([^"]+)"', right_section.group(1))
                    node_params['right_join_cols'] = right_cols
                # Extract join flags for determining join type
                for flag in ['includeMatchesInOutput', 'includeLeftUnmatchedInOutput', 'includeRightUnmatchedInOutput']:
                    m = re.search(rf'<entry\s+key="{flag}"\s+[^>]*value="([^"]*)"', content)
                    if m:
                        node_params[flag] = m.group(1) == 'true'
            
            if node_params:
                params_by_name[name] = node_params
                params_by_name[name_clean] = node_params
                # Also store by factory type for fallback matching
                if 'RuleEngineNodeFactory' in content:
                    params_by_name[f'_factory_rule_{name}'] = node_params
                    params_by_name['_has_rule_engine'] = True
                if 'GroupByNodeFactory' in content:
                    params_by_name[f'_factory_groupby_{name}'] = node_params
                    params_by_name['_has_groupby'] = True
                
                # Store by relative path for precise JSON matching
                if rel_path:
                    params_by_rel_path[rel_path] = node_params
                
        except Exception as e:
            log.add_warning(f"Error reading {settings_file.name}: {e}")
    
    # Also index by settings path (relative to extract_dir)
    params_by_path = {}
    for name, params in params_by_name.items():
        if isinstance(params, dict) and not name.startswith('_'):
            # Create path variants
            params_by_path[name] = params
    
    # Enrich nodes with extracted params
    enriched_count = 0
    for node in nodes:
        node_name = node.get('name', node.get('node_name', ''))
        node_factory = node.get('factory', '')
        settings_path = node.get('settings_path', '')
        matched = False
        
        # Priority 1: Match by settings_path node folder name (most precise)
        if settings_path:
            # Extract the node folder name (e.g., "Rule Engine (#1610)" from ".../Rule Engine (#1610)/settings.xml")
            norm_path = settings_path.replace('\\', '/')
            if norm_path.endswith('/settings.xml'):
                node_folder = norm_path.rsplit('/', 2)[-2]  # Get folder before settings.xml
            else:
                node_folder = norm_path.split('/')[-1]
            
            # Look for matching key in params_by_rel_path by node folder name
            for key in params_by_rel_path:
                # Extract node folder from stored key
                if key.endswith('/settings.xml'):
                    key_folder = key.rsplit('/', 2)[-2]
                else:
                    key_folder = key.split('/')[-1]
                
                if node_folder == key_folder:
                    node.update(params_by_rel_path[key])
                    enriched_count += 1
                    matched = True
                    break
        
        # Fallback 1: Exact name match
        if not matched and node_name in params_by_name:
            node.update(params_by_name[node_name])
            enriched_count += 1
            matched = True
        
        # Fallback 2: Partial name match
        if not matched:
            for param_name, params in params_by_name.items():
                if param_name.startswith('_'):  # Skip factory markers
                    continue
                if param_name in node_name or node_name in param_name:
                    if isinstance(params, dict):  # Ensure it's a params dict
                        node.update(params)
                        enriched_count += 1
                        matched = True
                        break
        
        # Fallback 3: Factory-based fallback for Rule Engine
        if not matched and 'RuleEngineNodeFactory' in node_factory:
            for param_name, params in params_by_name.items():
                if param_name.startswith('_factory_rule_') and isinstance(params, dict):
                    if params.get('python_rules'):
                        node.update(params)
                        enriched_count += 1
                        matched = True
                        break
        
        # Fallback 4: Factory-based fallback for GroupBy
        if not matched and 'GroupByNodeFactory' in node_factory:
            for param_name, params in params_by_name.items():
                if param_name.startswith('_factory_groupby_') and isinstance(params, dict):
                    if params.get('groupby_code'):
                        node.update(params)
                        enriched_count += 1
                        matched = True
                        break
    
    log.add_warning(f"Enriched {enriched_count} nodes with parameters")
    return nodes



def load_analysis_json(knwf_path, log):
    """Try to load pre-analyzed JSON."""
    paths_to_try = [
        knwf_path.parent / "workflow_analysis_v2.json",
        Path(__file__).parent / "workflow_analysis_v2.json",
    ]
    
    for path in paths_to_try:
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    nodes = data.get('nodes', [])
                    log.extraction_method = f"JSON ({path.name})"
                    return nodes
            except Exception as e:
                log.add_warning(f"Failed to load {path}: {e}")
    
    return None


def build_node_graph(nodes: list, connections: list) -> dict:
    """
    P2: Build node dependency graph from connections.
    
    Returns:
        dict with:
        - sorted_ids: list of node IDs in topological order
        - inputs: dict mapping node_id -> {port: source_node_id}
        - outputs: dict mapping node_id -> list of dest_node_ids
        - id_to_index: dict mapping node_id -> index in nodes list
    """
    if not connections:
        return {
            'sorted_ids': [str(i) for i in range(len(nodes))],
            'inputs': {},
            'outputs': {},
            'id_to_index': {str(i): i for i in range(len(nodes))}
        }
    
    # Build node_id -> index mapping
    id_to_index = {}
    for i, node in enumerate(nodes):
        node_id = node.get('node_id', node.get('id', str(i)))
        id_to_index[str(node_id)] = i
    
    # Build adjacency lists
    graph = {}  # node_id -> list of dest_node_ids
    in_degree = {}  # node_id -> count of incoming edges
    inputs = {}  # node_id -> {dest_port: source_node_id}
    
    # Initialize all nodes
    for node_id in id_to_index.keys():
        graph[node_id] = []
        in_degree[node_id] = 0
        inputs[node_id] = {}
    
    # Process connections
    for conn in connections:
        src = str(conn.get('source_id', ''))
        dst = str(conn.get('dest_id', ''))
        src_port = str(conn.get('source_port', '1'))
        dst_port = str(conn.get('dest_port', '1'))
        
        if src in graph and dst in in_degree:
            graph[src].append(dst)
            in_degree[dst] = in_degree.get(dst, 0) + 1
            inputs[dst][dst_port] = {'source': src, 'port': src_port}
    
    # Kahn's topological sort
    queue = [node_id for node_id, deg in in_degree.items() if deg == 0]
    sorted_ids = []
    
    while queue:
        node_id = queue.pop(0)
        sorted_ids.append(node_id)
        
        for neighbor in graph.get(node_id, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # If not all nodes sorted, there's a cycle - fallback to original order
    if len(sorted_ids) != len(id_to_index):
        sorted_ids = list(id_to_index.keys())
    
    return {
        'sorted_ids': sorted_ids,
        'inputs': inputs,
        'outputs': graph,
        'id_to_index': id_to_index
    }


# Template mappings (built-in, no external dependencies)
TEMPLATES = {
    # Column Operations
    'DataColumnSpecFilterNodeFactory': 'df = df.copy()  # Column Filter',
    'ColumnFilterNodeFactory': 'df = df.copy()  # Column Filter',
    'RenameNodeFactory': 'df = df.rename(columns={})  # Rename columns',
    'ColumnResorterNodeFactory': 'df = df[sorted_columns]  # Reorder columns',
    
    # Data Transformation
    'GroupByNodeFactory': 'df = df.groupby([]).agg({}).reset_index()',
    'SorterNodeFactory': 'df = df.sort_values(by=[])',
    'RowFilterNodeFactory': 'df = df[condition].copy()  # Row Filter',
    'DuplicateRowFilterNodeFactory': 'df = df.drop_duplicates()',
    
    # Joins — handled by dedicated code generators with port-aware wiring
    # (removed from TEMPLATES so they flow through the elif chain)
    # 'JoinerNodeFactory', 'Joiner3NodeFactory', 'CrossJoinerNodeFactory',
    # 'ConcatenateNodeFactory', 'AppendedRowsNodeFactory'
    
    # Math & Formulas
    'JEPNodeFactory': 'df["result"] = df["a"] + df["b"]  # Math Formula',
    'MathFormulaNodeFactory': 'df["result"] = df["a"] + df["b"]',
    'RoundDoubleNodeFactory': 'df["col"] = df["col"].round(2)',
    'ColumnAggregatorNodeFactory': 'result = df.agg("sum")',
    
    # String Operations
    'StringManipulationNodeFactory': 'df["col"] = df["col"].str.upper()',
    'CellSplitterNodeFactory': 'df = df["col"].str.split(",", expand=True)',
    'StringReplacerNodeFactory': 'df["col"] = df["col"].str.replace("old", "new")',
    
    # Type Conversion
    'NumberToString2NodeFactory': 'df["col"] = df["col"].astype(str)',
    'StringToNumber2NodeFactory': 'df["col"] = pd.to_numeric(df["col"], errors="coerce")',
    'DoubleToIntNodeFactory': 'df["col"] = df["col"].astype(int)',
    
    # Rule Engine (generic - use extracted rules when available)
    'RuleEngineNodeFactory': 'df["result"] = "TODO"  # Rule Engine - extract rules from settings.xml',
    'RuleEngineFilterNodeFactory': 'df = df.copy()  # Rule Filter - extract condition from settings.xml',
    
    # Flow Control — pass-through with comment
    'EmptyTableSwitchNodeFactory': 'df = df.copy()  # Empty Table Switch — pass-through',
    'EndifNodeFactory': 'df = df.copy()  # End IF — pass-through',
    'IfSwitchNodeFactory': 'df = df.copy()  # IF Switch — pass-through',
    
    # Loops — pass-through (iteration handled by KNIME runtime)
    'GroupLoopStartNodeFactory': 'df = df.copy()  # Group Loop Start — iteration managed externally',
    'LoopEndDynamicNodeFactory': 'df = df.copy()  # Loop End — pass-through',
    'LoopEndNodeFactory': 'df = df.copy()  # Loop End — pass-through',
    
    # Variables
    'TableToVariable3NodeFactory': 'flow_var = df.iloc[0].to_dict() if len(df) > 0 else {}  # Table to Variable',
    'VariableToTable4NodeFactory': 'df = pd.DataFrame([flow_vars]) if flow_vars else pd.DataFrame()',
    'ConstantValueColumnNodeFactory': 'df["constant"] = 1  # Constant Value Column',
    
    # Date/Time
    'OldToNewTimeNodeFactory': 'df["col"] = pd.to_datetime(df["col"])',
    'DateTimeDifferenceNodeFactory': 'df["diff"] = df["end"] - df["start"]',
    'CreateDateTimeNodeFactory': 'df["date"] = pd.Timestamp.now()',
    'CreateDateTimeRangeNodeFactory': 'df = pd.date_range(start, end, freq="D").to_frame()',
    'ExtractDateTimeFieldsNodeFactory': 'df["year"], df["month"], df["day"] = df["date"].dt.year, df["date"].dt.month, df["date"].dt.day',
    'ExtractDateTimeFieldsNodeFactory2': 'df["year"], df["month"], df["day"] = df["date"].dt.year, df["date"].dt.month, df["date"].dt.day',
    'DateTimeShiftNodeFactory': 'df["date"] = df["date"] + pd.Timedelta(days=1)',
    'ModifyTimeNodeFactory': 'df["time"] = df["time"].apply(modify_func)',
    
    # Missing Values
    'MissingValueNodeFactory': 'df = df.fillna(0)',
    
    # Database Connectors — removed from templates; handled by dedicated F4 code generators
    # 'JDBCConnectorNodeFactory', 'BigQueryDBConnectorNodeFactory', 'MySQLDBConnectorNodeFactory'
    # 'GoogleApiConnectorFactory', 'DBLoaderNodeFactory', 'DBLoaderNodeFactory2'
    'DatabaseLoopingNodeFactory': 'pass  # Database Loop - requires connection',
    
    # Database Readers — kept for fallback if SQL is not extracted
    'DBReaderNodeFactory': 'df = pd.read_sql(query, conn)',
    'DBQueryReaderNodeFactory': 'df = pd.read_sql(query, conn)',
    
    # Counter
    'CounterGenerationNodeFactory': 'df["counter"] = range(1, len(df) + 1)',
    
    # Add Empty Rows
    'AddEmptyRowsNodeFactory': 'df = pd.concat([df, pd.DataFrame([{}])])',
    
    # Row Splitter (outputs to two ports based on condition)
    'RowFilter2PortNodeFactory': 'df_match = df.copy()  # Row Splitter - extract filter condition from settings.xml',
    
    # Parameter Optimization Loops — pass-through
    'LoopStartParOptNodeFactory': 'df = df.copy()  # Parameter Optimization Loop Start — pass-through',
    'LoopEndParOptNodeFactory': 'df = df.copy()  # Parameter Optimization Loop End — pass-through',
    
    # Java Code (requires manual conversion)
    'JavaSnippetNodeFactory': 'pass  # TODO: Java Snippet requires manual conversion',
    'JavaEditVarNodeFactory': 'pass  # TODO: Java Edit Variable requires manual conversion',
    
    # Date/Time Converters
    'NewToOldTimeNodeFactory': 'df["col"] = df["col"].dt.to_pydatetime()  # DateTime to legacy',
}


def get_template(factory, log):
    """Get Python template for a KNIME factory."""
    for key, template in TEMPLATES.items():
        if key in factory:
            return template, key.replace('NodeFactory', '')
    return None, ""


def generate_code(nodes, log, graph=None):
    """Generate Python code from nodes with embedded SQL queries.
    
    Args:
        nodes: List of node dicts with factory, name, params.
        log: TranspilerLog instance.
        graph: Optional dict from build_node_graph() with sorted_ids, inputs, id_to_index.
    """
    lines = [
        '"""',
        'Auto-generated Python pipeline from KNIME workflow',
        f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        f'Nodes: {len(nodes)}',
        '',
        'This code is 100% faithful to the original KNIME workflow.',
        'SQL queries are embedded inline from Database Reader nodes.',
        'Variables use df_<node_id> naming for multi-input wiring.',
        '"""',
        'import pandas as pd',
        'import numpy as np',
        'from datetime import datetime, timedelta',
        '',
    ]
    
    # F4: Generate DB connection header from extracted connector nodes
    # Only include actual connector/authentication nodes (not readers/loaders)
    _JDBC_FAMILY = {'jdbc', 'sybase', 'mysql', 'postgres', 'oracle', 'mssql',
                    'hive', 'snowflake', 'redshift', 'sqlite', 'h2', 'db2',
                    'teradata', 'vertica', 'mariadb'}
    _CONNECTOR_TYPES = _JDBC_FAMILY | {'bigquery', 'google_api'}
    
    db_connectors = []
    for node in nodes:
        if node.get('is_connector'):
            db_connectors.append(node)
    
    if db_connectors:
        lines.extend([
            '',
            '# ============================================================',
            '# DATABASE CONNECTIONS (extracted from KNIME connector nodes)',
            '# ============================================================',
            '# ⚠️  FILL IN your credentials below before running.',
            '# ============================================================',
            '',
        ])
        
        seen_types = set()
        for node in db_connectors:
            db_type = node.get('db_type', '')
            node_name = node.get('name', node.get('node_name', ''))
            
            if db_type in _JDBC_FAMILY and 'jdbc' not in seen_types:
                seen_types.add('jdbc')
                jdbc_url = node.get('jdbc_url', 'jdbc:driver://host:port/database')
                jdbc_driver = node.get('jdbc_driver', 'com.example.jdbc.Driver')
                
                # Determine Python driver recommendation
                py_driver = 'jaydebeapi'
                pip_extra = ''
                if 'sybase' in jdbc_driver.lower() or 'sybase' in jdbc_url.lower():
                    pip_extra = '  # pip install jaydebeapi JPype1'
                elif 'mysql' in jdbc_driver.lower():
                    py_driver = 'mysql.connector'
                    pip_extra = '  # pip install mysql-connector-python'
                elif 'postgresql' in jdbc_driver.lower() or 'postgres' in jdbc_url.lower():
                    py_driver = 'psycopg2'
                    pip_extra = '  # pip install psycopg2-binary'
                elif 'sqlserver' in jdbc_driver.lower() or 'mssql' in jdbc_url.lower():
                    py_driver = 'pyodbc'
                    pip_extra = '  # pip install pyodbc'
                elif 'oracle' in jdbc_driver.lower():
                    py_driver = 'cx_Oracle'
                    pip_extra = '  # pip install cx_Oracle'
                elif 'hive' in jdbc_driver.lower():
                    py_driver = 'pyhive'
                    pip_extra = '  # pip install pyhive[hive]'
                else:
                    pip_extra = '  # pip install jaydebeapi JPype1'
                
                lines.append(f'# --- JDBC Connection: {node_name} ---')
                lines.append(f'# KNIME Driver: {jdbc_driver}')
                lines.append(f'# KNIME JDBC URL: {jdbc_url}')
                lines.append(f'import jaydebeapi{pip_extra}')
                lines.append(f'')
                lines.append(f'JDBC_DRIVER = "{jdbc_driver}"')
                lines.append(f'JDBC_URL = "{jdbc_url}"')
                lines.append(f'JDBC_USER = ""      # ← PREENCHA seu usuário')
                lines.append(f'JDBC_PASSWORD = ""   # ← PREENCHA sua senha')
                lines.append(f'# JDBC_JAR = "path/to/driver.jar"  # ← caminho do .jar do driver')
                lines.append(f'')
                lines.append(f'def get_jdbc_connection():')
                lines.append(f'    """Establish JDBC connection (Sybase/Hive/etc)."""')
                lines.append(f'    return jaydebeapi.connect(')
                lines.append(f'        JDBC_DRIVER,')
                lines.append(f'        JDBC_URL,')
                lines.append(f'        [JDBC_USER, JDBC_PASSWORD],')
                lines.append(f'        # JDBC_JAR,  # ← descomente se necessário')
                lines.append(f'    )')
                lines.append(f'')
            
            if db_type == 'google_api' and 'google_api' not in seen_types:
                seen_types.add('google_api')
                sa_email = node.get('service_account', '')
                lines.append(f'# --- Google API Authentication: {node_name} ---')
                if sa_email:
                    lines.append(f'# KNIME Service Account: {sa_email}')
                lines.append(f'from google.oauth2 import service_account  # pip install google-auth')
                lines.append(f'')
                lines.append(f'GOOGLE_KEY_FILE = ""  # ← PREENCHA o caminho do arquivo .json da service account')
                lines.append(f'GOOGLE_SCOPES = ["https://www.googleapis.com/auth/bigquery"]')
                lines.append(f'')
                lines.append(f'def get_google_credentials():')
                lines.append(f'    """Load Google service account credentials."""')
                lines.append(f'    return service_account.Credentials.from_service_account_file(')
                lines.append(f'        GOOGLE_KEY_FILE, scopes=GOOGLE_SCOPES')
                lines.append(f'    )')
                lines.append(f'')
            
            if db_type == 'bigquery' and 'bigquery' not in seen_types:
                seen_types.add('bigquery')
                bq_project = node.get('bq_project', '')
                # Fallback: scan all nodes for bq_project_from_sql
                if not bq_project:
                    for n in nodes:
                        bq_project = n.get('bq_project_from_sql', '')
                        if bq_project:
                            break
                if not bq_project:
                    bq_project = 'your-project-id'
                lines.append(f'# --- BigQuery Connection: {node_name} ---')
                lines.append(f'from google.cloud import bigquery  # pip install google-cloud-bigquery')
                lines.append(f'')
                lines.append(f'BQ_PROJECT = "{bq_project}"')
                lines.append(f'')
                lines.append(f'def get_bigquery_client():')
                lines.append(f'    """Create BigQuery client."""')
                lines.append(f'    credentials = get_google_credentials()')
                lines.append(f'    return bigquery.Client(project=BQ_PROJECT, credentials=credentials)')
                lines.append(f'')
        
        lines.append('')
    
    lines.extend([
        '',
        '# ============================================================',
        '# SQL QUERY DEFINITIONS (extracted from KNIME nodes)',
        '# ============================================================',
        '',
    ])
    
    # Build helper for variable naming
    # Use _uid directly for guaranteed-unique variable names.
    # _uid is the sequential JSON index (0, 1, 2...) assigned
    # in load_analysis_json, ensuring zero naming collisions.
    
    def var_for(uid):
        """Return unique df_ variable name from a _uid."""
        return f'df_{str(uid).replace("-", "_")}'
    
    def get_upstream_var(uid, port='1'):
        """Get the variable name of the upstream node."""
        if not graph or uid not in graph.get('inputs', {}):
            return 'df'
        port_info = graph['inputs'][uid].get(str(port), {})
        if port_info:
            return var_for(port_info['source'])
        # Fallback: try first available port
        for p_key, p_val in graph['inputs'][uid].items():
            return var_for(p_val['source'])
        return 'df'
    
    def get_all_upstream_vars(uid):
        """Get all upstream variable names ordered by dest port."""
        if not graph or uid not in graph.get('inputs', {}):
            return ['df']
        port_map = graph['inputs'][uid]
        if not port_map:
            return ['df']
        sorted_ports = sorted(
            port_map.keys(),
            key=lambda x: int(x) if x.isdigit() else 0)
        return [var_for(port_map[p]['source']) for p in sorted_ports]
    
    # Determine node ordering: use topological sort if available
    # With _uid, ALL 299 nodes have unique keys — no dedup/fallback needed
    if graph and graph.get('sorted_ids'):
        ordered_indices = []
        for uid in graph['sorted_ids']:
            idx = graph['id_to_index'].get(uid)
            if idx is not None and idx < len(nodes):
                ordered_indices.append((idx, uid))
    else:
        ordered_indices = [(i, str(i)) for i in range(len(nodes))]
    
    # First pass: generate SQL query definitions for DB nodes
    sql_node_ids = []
    for i, node in enumerate(nodes):
        factory = node.get('factory', '')
        name = node.get('name', node.get('node_name', f'Node_{i}'))
        sql = node.get('sql_query', '')
        
        if sql and any(p in factory for p in ['DBReader', 'DBQueryReader', 'DatabaseLooping']):
            func_name = f"sql_{i}_{re.sub(r'[^a-zA-Z0-9_]', '_', name)[:30]}"
            sql_node_ids.append((i, func_name))
            
            lines.append(f'def {func_name}(conn, **params):')
            lines.append(f'    """')
            lines.append(f'    {name}')
            lines.append(f'    Node ID: {node.get("id", i)}')
            lines.append(f'    """')
            lines.append(f'    sql = """')
            
            for sql_line in sql.split('\n'):
                lines.append(f'        {sql_line}')
            
            lines.append(f'    """')
            lines.append(f'    for key, val in params.items():')
            lines.append(f'        sql = sql.replace(f"$$${{{{key}}}}$$", str(val))')
            lines.append(f'    return pd.read_sql(sql, conn)')
            lines.append('')
    
    lines.extend([
        '',
        '# ============================================================',
        '# MAIN PIPELINE FUNCTION',
        '# ============================================================',
        '',
        'def run_pipeline(conn=None, df_input: pd.DataFrame = None, **flow_vars) -> pd.DataFrame:',
        '    """',
        '    Execute the transpiled KNIME workflow.',
        '    ',
        '    Args:',
        '        conn: Database connection (required for SQL nodes)',
        '        df_input: Optional input DataFrame',
        '        **flow_vars: KNIME flow variables (e.g. SDtInicialMesAnt, SDtFinalMesAnt)',
        '    ',
        '    Flow Variable Propagation:',
        '        Table Row to Variable nodes extract first-row values as flow_vars.',
        '        These are injected into SQL queries via $${VAR_NAME}$$ substitution.',
        '        Pass custom values to override computed dates for re-runs.',
        '    ',
        '    Returns:',
        '        pd.DataFrame: Result of the pipeline',
        '    """',
        '    df = df_input.copy() if df_input is not None else pd.DataFrame()',
        '',
    ])
    
    # Second pass: generate pipeline steps in topological order
    sql_func_map = {i: func for i, func in sql_node_ids}
    last_var = 'df'  # Track last assigned variable for final return
    
    for idx, uid in ordered_indices:
        node = nodes[idx]
        i = idx
        factory = node.get('factory', '')
        name = node.get('name', node.get('node_name', f'Node_{i}'))
        sql = node.get('sql_query', '')
        out_var = var_for(uid)
        upstream = get_all_upstream_vars(uid)
        in_var = upstream[0] if upstream else 'df'
        
        # Check if this node has a SQL function defined
        if i in sql_func_map:
            func_name = sql_func_map[i]
            log.add_node(name, factory, True, "SQLQuery")
            lines.append(f'    # {name}')
            lines.append(f'    {out_var} = {func_name}(conn, **flow_vars)')
            last_var = out_var
            lines.append('')
            continue
        
        # Check for extracted parameters first
        generated = False
        
        # Math Formula with extracted expression
        if node.get('expression'):
            expr = node['expression'].replace('df[', f'{in_var}[')
            target = node.get('target_column', 'result')
            log.add_node(name, factory, True, "MathFormula_Extracted")
            lines.append(f'    # {name}')
            lines.append(f'    {out_var} = {in_var}.copy()')
            lines.append(f'    {out_var}["{target}"] = {expr}')
            last_var = out_var
            lines.append('')
            generated = True
        
        # Rule Engine with extracted rules
        elif node.get('python_rules'):
            rules_code = node['python_rules']
            output_col = node.get('output_column', 'result')
            log.add_node(name, factory, True, "RuleEngine_Extracted")
            lines.append(f'    # {name}')
            lines.append(f'    {out_var} = {in_var}.copy()')
            for rule_line in rules_code.split('\n'):
                rule_line = rule_line.replace('df["result"]', f'{out_var}["{output_col}"]')
                rule_line = rule_line.replace('df["', f'{out_var}["')
                lines.append(f'    {rule_line}')
            last_var = out_var
            lines.append('')
            generated = True
        
        # GroupBy with pre-computed code (P1: Enhanced)
        elif node.get('groupby_code'):
            log.add_node(name, factory, True, "GroupBy_P1")
            code = node['groupby_code'].replace('df = df.', f'{out_var} = {in_var}.')
            lines.append(f'    # {name}')
            lines.append(f'    {code}')
            last_var = out_var
            lines.append('')
            generated = True
        
        # GroupBy with extracted columns and aggregations (fallback)
        elif node.get('group_columns') or node.get('aggregations'):
            grp_cols = node.get('group_columns', [])
            aggs = node.get('aggregations', {})
            log.add_node(name, factory, True, "GroupBy_Extracted")
            lines.append(f'    # {name}')
            
            if grp_cols and isinstance(aggs, dict) and aggs:
                agg_str = ', '.join(f'"{c}": "{m}"' for c, m in aggs.items())
                grp_str = ', '.join(f'"{c}"' for c in grp_cols)
                lines.append(f'    {out_var} = {in_var}.groupby([{grp_str}]).agg({{{agg_str}}}).reset_index()')
            elif grp_cols:
                grp_str = ', '.join(f'"{c}"' for c in grp_cols)
                lines.append(f'    {out_var} = {in_var}.groupby([{grp_str}]).first().reset_index()')
            else:
                lines.append(f'    {out_var} = {in_var}.copy()  # GroupBy: no config')
            last_var = out_var
            lines.append('')
            generated = True
        
        # F2.2: Joiner with extracted join columns + port-aware wiring
        elif 'JoinerNodeFactory' in factory and 'Cross' not in factory:
            left_cols = node.get('left_join_cols', [])
            right_cols = node.get('right_join_cols', [])
            include_match = node.get('includeMatchesInOutput', True)
            include_left = node.get('includeLeftUnmatchedInOutput', False)
            include_right = node.get('includeRightUnmatchedInOutput', False)
            if include_match and include_left and include_right:
                how = 'outer'
            elif include_match and include_left:
                how = 'left'
            elif include_match and include_right:
                how = 'right'
            else:
                how = 'inner'
            # Wire from upstream ports: port 1 = left, port 2 = right
            df_left = upstream[0] if len(upstream) > 0 else 'df'
            df_right = upstream[1] if len(upstream) > 1 else df_left
            log.add_node(name, factory, True, "Joiner_Wired")
            lines.append(f'    # {name}')
            if left_cols and right_cols and left_cols == right_cols:
                cols_str = ', '.join(f'"{c}"' for c in left_cols)
                lines.append(f'    {out_var} = {df_left}.merge({df_right}, on=[{cols_str}], how="{how}")')
            elif left_cols and right_cols:
                left_str = ', '.join(f'"{c}"' for c in left_cols)
                right_str = ', '.join(f'"{c}"' for c in right_cols)
                lines.append(f'    {out_var} = {df_left}.merge({df_right}, left_on=[{left_str}], right_on=[{right_str}], how="{how}")')
            else:
                lines.append(f'    {out_var} = {df_left}.merge({df_right}, how="{how}")')
            last_var = out_var
            lines.append('')
            generated = True
        
        # F2.4: Cross Joiner with port-aware wiring
        elif 'CrossJoinerNodeFactory' in factory:
            df_left = upstream[0] if len(upstream) > 0 else 'df'
            df_right = upstream[1] if len(upstream) > 1 else df_left
            log.add_node(name, factory, True, "CrossJoiner_Wired")
            lines.append(f'    # {name}')
            lines.append(f'    {out_var} = {df_left}.merge({df_right}, how="cross")')
            last_var = out_var
            lines.append('')
            generated = True
        
        # F2.3: Concatenate / AppendedRows with N-input wiring
        elif 'ConcatenateNodeFactory' in factory or 'AppendedRowsNodeFactory' in factory:
            concat_vars = upstream if len(upstream) > 1 else [in_var]
            vars_str = ', '.join(concat_vars)
            log.add_node(name, factory, True, "Concatenate_Wired")
            lines.append(f'    # {name}')
            lines.append(f'    {out_var} = pd.concat([{vars_str}], ignore_index=True)')
            last_var = out_var
            lines.append('')
            generated = True
        
        # F4: DB Connector nodes — generate connection setup calls
        elif 'JDBCConnectorNodeFactory' in factory or 'DatabaseConnectorNodeFactory' in factory:
            jdbc_url = node.get('jdbc_url', '')
            jdbc_driver = node.get('jdbc_driver', '')
            log.add_node(name, factory, True, "JDBC_Connector")
            lines.append(f'    # {name}')
            lines.append(f'    # JDBC: {jdbc_url}' if jdbc_url else f'    # JDBC Connector')
            lines.append(f'    conn_jdbc = get_jdbc_connection()')
            last_var = out_var
            lines.append('')
            generated = True
        
        elif 'GoogleApiConnectorFactory' in factory:
            sa = node.get('service_account', '')
            log.add_node(name, factory, True, "GoogleApi_Connector")
            lines.append(f'    # {name}')
            if sa:
                lines.append(f'    # Service Account: {sa}')
            lines.append(f'    credentials = get_google_credentials()')
            last_var = out_var
            lines.append('')
            generated = True
        
        elif 'BigQueryDBConnectorNodeFactory' in factory:
            bq_proj = node.get('bq_project', '')
            log.add_node(name, factory, True, "BigQuery_Connector")
            lines.append(f'    # {name}')
            if bq_proj:
                lines.append(f'    # Project: {bq_proj}')
            lines.append(f'    bq_client = get_bigquery_client()')
            last_var = out_var
            lines.append('')
            generated = True
        
        elif 'MySQLDBConnectorNodeFactory' in factory:
            log.add_node(name, factory, True, "MySQL_Connector")
            lines.append(f'    # {name}')
            lines.append(f'    # conn_mysql = mysql.connector.connect(host="", user="", password="", database="")')
            last_var = out_var
            lines.append('')
            generated = True
        
        # F4: DBLoader — emit to_sql/to_gbq with extracted table/schema
        elif 'DBLoaderNodeFactory' in factory:
            table = node.get('db_table', 'table')
            schema = node.get('db_schema', '')
            full_table = f'{schema}.{table}' if schema else table
            log.add_node(name, factory, True, "DBLoader_Extracted")
            lines.append(f'    # {name}')
            lines.append(f'    # Load DataFrame → {full_table}')
            lines.append(f'    {in_var}.to_gbq("{full_table}", project_id=BQ_PROJECT, if_exists="append", credentials=credentials)')
            last_var = out_var
            lines.append('')
            generated = True
        
        # F4: Python Script nodes — prevent template fallback
        elif 'PythonScriptNodeFactory' in factory or 'Python2ScriptNodeFactory' in factory:
            log.add_node(name, factory, True, "PythonScript")
            lines.append(f'    # {name} — Python Script (embedded code below)')
            last_var = out_var
            lines.append('')
            generated = True
        
        # Column Filter with extracted columns
        elif node.get('included_columns'):
            cols = node['included_columns']
            log.add_node(name, factory, True, "ColumnFilter_Extracted")
            lines.append(f'    # {name}')
            cols_str = ', '.join(f'"{c}"' for c in cols)
            lines.append(f'    {out_var} = {in_var}[[{cols_str}]]')
            last_var = out_var
            lines.append('')
            generated = True
        
        # Column Rename with extracted mappings
        elif node.get('rename_map'):
            rename_map = node['rename_map']
            log.add_node(name, factory, True, "Rename_Extracted")
            lines.append(f'    # {name}')
            rename_str = ', '.join(f'"{old}": "{new}"' for old, new in rename_map.items())
            lines.append(f'    {out_var} = {in_var}.rename(columns={{{rename_str}}})')
            last_var = out_var
            lines.append('')
            generated = True
        
        # Row Filter with extracted filter code (F1.5)
        elif node.get('filter_code'):
            filter_code = node['filter_code'].replace('df[', f'{in_var}[').replace('df =', f'{out_var} =')
            log.add_node(name, factory, True, "RowFilter_Extracted")
            lines.append(f'    # {name}')
            lines.append(f'    {filter_code}')
            last_var = out_var
            lines.append('')
            generated = True
        
        # Sorter with extracted columns and direction (F1.4)
        elif node.get('sort_columns'):
            sort_cols = node['sort_columns']
            ascending = node.get('sort_ascending', [True] * len(sort_cols))
            log.add_node(name, factory, True, "Sorter_Extracted")
            lines.append(f'    # {name}')
            cols_str = ', '.join(f'"{c}"' for c in sort_cols)
            asc_str = ', '.join(str(a) for a in ascending)
            lines.append(f'    {out_var} = {in_var}.sort_values(by=[{cols_str}], ascending=[{asc_str}]).reset_index(drop=True)')
            last_var = out_var
            lines.append('')
            generated = True
        
        # F3.1: String Manipulation with extracted expression
        elif node.get('string_expr'):
            orig = node.get('original_expression', '')
            target = node.get('target_column', '')
            lines.append(f'    # {name}')
            lines.append(f'    {out_var} = {in_var}.copy()')
            
            # Tier 1: Deterministic pattern-based converter
            det_code = ''
            if target and orig:
                det_code = translate_knime_string_expr(orig, target, out_var)
            
            if det_code:
                log.add_node(name, factory, True, "StringManip_Deterministic")
                for code_line in det_code.split('\n'):
                    lines.append(f'    {code_line}')
            else:
                # Tier 2: LLM translation
                llm_code = ''
                if target and orig and LLM_STRING_AVAILABLE:
                    llm_code = llm_translate_string(orig, target, out_var)
                
                if llm_code:
                    log.add_node(name, factory, True, "StringManip_LLM")
                    for code_line in llm_code.split('\n'):
                        lines.append(f'    {code_line}')
                else:
                    # Tier 3: Comment fallback
                    log.add_node(name, factory, True, "StringManip_Extracted")
                    for expr_line in orig.split('\n'):
                        lines.append(f'    # KNIME: {expr_line.strip()[:80]}')
                    if target:
                        lines.append(f'    # Target column: {target}')
                        lines.append(f'    # TODO: Translate String Manipulation to pandas')
            
            last_var = out_var
            lines.append('')
            generated = True
        
        # F3.2: Rule Engine Filter (row filtering based on rules)
        elif node.get('filter_rules'):
            rules = node['filter_rules']
            log.add_node(name, factory, True, "RuleFilter_Translated")
            lines.append(f'    # {name} — Rule Engine Filter')
            # Translate rules to pandas boolean condition
            condition = translate_filter_rules_to_pandas(rules, out_var)
            if condition:
                lines.append(f'    {out_var} = {in_var}.copy()')
                lines.append(f'    {out_var} = {out_var}[{condition}]')
            else:
                lines.append(f'    {out_var} = {in_var}.copy()')
                for rule_line in rules.split('\n')[:3]:
                    lines.append(f'    # Rule: {rule_line.strip()[:70]}')
            last_var = out_var
            lines.append('')
            generated = True
        
        # F3.3: Constant Value Column
        elif node.get('const_column') or node.get('const_value') is not None:
            col = node.get('const_column', 'new_column')
            val = node.get('const_value', '')
            ctype = node.get('const_type', 'StringValue')
            log.add_node(name, factory, True, "ConstValue_Extracted")
            lines.append(f'    # {name}')
            lines.append(f'    {out_var} = {in_var}.copy()')
            if 'Int' in ctype or 'Long' in ctype:
                lines.append(f'    {out_var}["{col}"] = {val}')
            elif 'Double' in ctype:
                lines.append(f'    {out_var}["{col}"] = {val}')
            else:
                lines.append(f'    {out_var}["{col}"] = "{val}"')
            last_var = out_var
            lines.append('')
            generated = True
        
        # F3.4: Date/Time transformations
        elif node.get('date_action'):
            action = node['date_action']
            log.add_node(name, factory, True, f"DateTime_{action}")
            lines.append(f'    # {name}')
            lines.append(f'    {out_var} = {in_var}.copy()')
            
            if action == 'old_to_new':
                cols = node.get('date_columns', [])
                col = node.get('date_column', '')
                if cols:
                    for c in cols:
                        lines.append(f'    {out_var}["{c}"] = pd.to_datetime({out_var}["{c}"], errors="coerce")')
                elif col:
                    lines.append(f'    {out_var}["{col}"] = pd.to_datetime({out_var}["{col}"], errors="coerce")')
                else:
                    lines.append(f'    # OldToNewTime: column not extracted')
            elif action == 'new_to_old':
                cols = node.get('date_columns', [])
                col = node.get('date_column', '')
                if cols:
                    for c in cols:
                        lines.append(f'    {out_var}["{c}"] = {out_var}["{c}"].dt.strftime("%Y-%m-%d")')
                elif col:
                    lines.append(f'    {out_var}["{col}"] = {out_var}["{col}"].dt.strftime("%Y-%m-%d")')
                else:
                    lines.append(f'    # NewToOldTime: column not extracted')
            elif action == 'difference':
                col1 = node.get('col1', node.get('first_column', ''))
                col2 = node.get('col2', node.get('second_column', ''))
                out_col = node.get('output_column', 'date_diff')
                gran = node.get('granularity', 'DAYS').upper()
                if col1 and col2:
                    if 'DAY' in gran:
                        lines.append(f'    {out_var}["{out_col}"] = (pd.to_datetime({out_var}["{col1}"]) - pd.to_datetime({out_var}["{col2}"])).dt.days')
                    elif 'HOUR' in gran:
                        lines.append(f'    {out_var}["{out_col}"] = (pd.to_datetime({out_var}["{col1}"]) - pd.to_datetime({out_var}["{col2}"])).dt.total_seconds() / 3600')
                    else:
                        lines.append(f'    {out_var}["{out_col}"] = pd.to_datetime({out_var}["{col1}"]) - pd.to_datetime({out_var}["{col2}"])')
                else:
                    lines.append(f'    # DateTimeDifference: columns not extracted')
            elif action == 'extract_fields':
                lines.append(f'    # ExtractDateTimeFields: extract year/month/day/hour from datetime columns')
            elif action == 'shift':
                lines.append(f'    # DateTimeShift: shift datetime by period')
            elif action == 'create':
                lines.append(f'    # CreateDateTime: create datetime column')
            elif action == 'modify':
                lines.append(f'    # ModifyTime: modify time component')
            else:
                lines.append(f'    # DateTime action: {action}')
            
            last_var = out_var
            lines.append('')
            generated = True
        
        # F3.5: RoundDouble
        elif node.get('precision') is not None or node.get('round_column'):
            precision = node.get('precision', 2)
            col = node.get('round_column', '')
            log.add_node(name, factory, True, "Round_Extracted")
            lines.append(f'    # {name}')
            lines.append(f'    {out_var} = {in_var}.copy()')
            if col:
                lines.append(f'    {out_var}["{col}"] = {out_var}["{col}"].round({precision})')
            else:
                lines.append(f'    {out_var} = {out_var}.round({precision})')
            last_var = out_var
            lines.append('')
            generated = True
        
        # F3.6: TableToVariable
        elif node.get('action') == 'table_to_variable':
            log.add_node(name, factory, True, "TableToVar")
            lines.append(f'    # {name} — Table Row to Variable')
            lines.append(f'    # Propagates first-row values as flow variables to downstream SQL nodes')
            lines.append(f'    {out_var} = {in_var}.copy()')
            lines.append(f'    if not {in_var}.empty:')
            lines.append(f'        flow_vars.update({in_var}.iloc[0].to_dict())')
            last_var = out_var
            lines.append('')
            generated = True
        
        # F3.7: NumberToString
        elif node.get('action') == 'number_to_string':
            cols = node.get('convert_columns', [])
            log.add_node(name, factory, True, "NumToStr_Extracted")
            lines.append(f'    # {name}')
            lines.append(f'    {out_var} = {in_var}.copy()')
            if cols:
                for col in cols:
                    lines.append(f'    {out_var}["{col}"] = {out_var}["{col}"].astype(str)')
            else:
                lines.append(f'    # NumberToString: columns not extracted')
            last_var = out_var
            lines.append('')
            generated = True
        
        # F3.7b: StringToNumber
        elif node.get('action') == 'string_to_number':
            cols = node.get('convert_columns', [])
            log.add_node(name, factory, True, "StrToNum_Extracted")
            lines.append(f'    # {name}')
            lines.append(f'    {out_var} = {in_var}.copy()')
            if cols:
                for col in cols:
                    lines.append(f'    {out_var}["{col}"] = pd.to_numeric({out_var}["{col}"], errors="coerce")')
            else:
                lines.append(f'    # StringToNumber: columns not extracted')
            last_var = out_var
            lines.append('')
            generated = True
        
        # F3.7c: DoubleToInt
        elif node.get('action') == 'double_to_int':
            cols = node.get('convert_columns', [])
            log.add_node(name, factory, True, "DblToInt_Extracted")
            lines.append(f'    # {name}')
            lines.append(f'    {out_var} = {in_var}.copy()')
            if cols:
                for col in cols:
                    lines.append(f'    {out_var}["{col}"] = {out_var}["{col}"].astype("Int64")')
            else:
                lines.append(f'    # DoubleToInt: columns not extracted')
            last_var = out_var
            lines.append('')
            generated = True
        
        # F3.8: VariableToTable (fix: avoids df.pd.DataFrame syntax error)
        elif node.get('action') == 'variable_to_table':
            log.add_node(name, factory, True, "VarToTable_Extracted")
            lines.append(f'    # {name}')
            lines.append(f'    {out_var} = pd.DataFrame([flow_vars]) if flow_vars else pd.DataFrame()')
            last_var = out_var
            lines.append('')
            generated = True
        
        # F3.9: ColumnResorter (fix: avoids undefined sorted_columns)
        elif node.get('action') == 'column_resorter':
            cols = node.get('resorter_columns', [])
            log.add_node(name, factory, True, "ColResorter_Extracted")
            lines.append(f'    # {name}')
            if cols:
                cols_str = ', '.join(f'"{c}"' for c in cols)
                lines.append(f'    {out_var} = {in_var}[[{cols_str}]]')
            else:
                lines.append(f'    {out_var} = {in_var}.copy()  # ColumnResorter: order not extracted')
            last_var = out_var
            lines.append('')
            generated = True
        
        # F3.10: RowSplitter / RowFilter2Port (fix: avoids empty condition df[""] == "")
        elif node.get('action') == 'row_splitter':
            filter_col = node.get('filter_column', '')
            filter_op = node.get('filter_operator', '')
            filter_val = node.get('filter_value', '')
            log.add_node(name, factory, True, "RowSplitter_Extracted")
            lines.append(f'    # {name}')
            lines.append(f'    {out_var} = {in_var}.copy()')
            if filter_col and filter_op:
                op_map = {
                    'EQ': '==', 'NEQ': '!=', 'LT': '<', 'LTE': '<=',
                    'GT': '>', 'GTE': '>=', 'LIKE': '.str.contains',
                }
                pandas_op = op_map.get(filter_op, '==')
                if 'contains' in pandas_op:
                    lines.append(f'    {out_var} = {out_var}[{out_var}["{filter_col}"].str.contains("{filter_val}", na=False)]')
                else:
                    lines.append(f'    {out_var} = {out_var}[{out_var}["{filter_col}"] {pandas_op} "{filter_val}"]')
            elif filter_col:
                lines.append(f'    {out_var} = {out_var}[{out_var}["{filter_col}"].notna()]')
            else:
                lines.append(f'    # RowSplitter: condition not extracted')
            last_var = out_var
            lines.append('')
            generated = True
        
        # F7: Python2Script — inject extracted Python code with variable remapping
        elif node.get('python_source') and not generated:
            py_src = node['python_source']
            
            lines.append(f'    # {name} — Python Script (extracted from KNIME)')
            lines.append(f'    {out_var} = {in_var}.copy()')
            
            # Remap KNIME Python variables to transpiler variables
            # KNIME uses input_table_1/output_table_1 convention
            remapped = py_src
            remapped = remapped.replace('input_table_1', out_var)
            remapped = remapped.replace('input_table_2', out_var)
            remapped = remapped.replace('output_table_1', out_var)
            remapped = remapped.replace('output_table_2', out_var)
            
            # Validate AST of remapped code
            try:
                import ast as _ast
                _ast.parse(remapped)
                log.add_node(name, factory, True, "PythonScript_Injected")
                for code_line in remapped.split('\n'):
                    lines.append(f'    {code_line}')
            except SyntaxError:
                # AST failed — inject as comments for manual review
                log.add_node(name, factory, True, "PythonScript_Extracted")
                lines.append(f'    # Python source code (AST validation failed):')
                for code_line in py_src.split('\n')[:20]:
                    lines.append(f'    # {code_line}')
            
            last_var = out_var
            lines.append('')
            generated = True
        
        # F6: Java Snippet / Java Edit Variable — deterministic + LLM
        elif node.get('java_body') and not generated:
            java_body = node['java_body']
            java_in = node.get('java_in_cols', [])
            java_out = node.get('java_out_cols', [])
            
            lines.append(f'    # {name}')
            lines.append(f'    {out_var} = {in_var}.copy()')
            
            # Tier 1: Deterministic if/else → np.where() converter
            det_code = translate_java_body(java_body, java_in, java_out, out_var)
            
            if det_code:
                log.add_node(name, factory, True, "JavaSnippet_Deterministic")
                for code_line in det_code.split('\n'):
                    lines.append(f'    {code_line}')
            else:
                # Tier 2: LLM Java→pandas translation
                llm_code = ''
                if LLM_STRING_AVAILABLE:
                    llm_code = llm_translate_java(java_body, java_in, java_out, out_var)
                
                if llm_code:
                    log.add_node(name, factory, True, "JavaSnippet_LLM")
                    for code_line in llm_code.split('\n'):
                        lines.append(f'    {code_line}')
                else:
                    # Tier 3: Comment fallback
                    log.add_node(name, factory, True, "JavaSnippet_Extracted")
                    lines.append(f'    # Java code (needs manual conversion):')
                    for jl in java_body.split('\n')[:10]:
                        lines.append(f'    # {jl.strip()[:80]}')
                    lines.append(f'    # TODO: Java Snippet requires manual conversion')
            
            last_var = out_var
            lines.append('')
            generated = True
        
        # Fall back to template
        if not generated:
            template, template_name = get_template(factory, log)
            
            if template:
                # Wire template to use per-node variables
                # Split on first ' = ' to handle LHS (output) and RHS (input) separately
                if ' = ' in template:
                    lhs, rhs = template.split(' = ', 1)
                    # LHS: replace 'df' with out_var (e.g. df → df_4)
                    wired_lhs = lhs.replace('df', out_var, 1)
                    # RHS: replace 'df.' / 'df[' / 'df,' / 'df)' with in_var refs
                    import re as _re
                    wired_rhs = _re.sub(r'\bdf\b', in_var, rhs)
                    wired = f'{wired_lhs} = {wired_rhs}'
                else:
                    wired = template
                if wired == template:  # No df reference in template (e.g. pass)
                    wired = template
                    lines.append(f'    # {name}')
                    lines.append(f'    {out_var} = {in_var}.copy()' if 'pass' not in template else f'    {template}')
                else:
                    lines.append(f'    # {name}')
                    lines.append(f'    {wired}')
                last_var = out_var
                log.add_node(name, factory, True, template_name)
                lines.append('')
            else:
                # Try LLM fallback for unknown nodes
                node_settings = {
                    k: v for k, v in node.items() 
                    if k not in ('name', 'factory', 'id')
                }
                llm_code, was_llm = llm_fallback(factory, name, node_settings)
                
                if was_llm:
                    # LLM generated code
                    log.add_node(name, factory, True, "LLM_Generated")
                    log.add_warning(f"LLM generated code for: {name}")
                    lines.append(f'    # {name} (LLM)')
                    for code_line in llm_code.split('\n'):
                        lines.append(f'    {code_line}')
                    lines.append('')
                else:
                    # Fallback to TODO
                    log.add_node(name, factory, False)
                    simple = factory.split('.')[-1].replace('NodeFactory', '') if factory else 'Unknown'
                    lines.append(f'    # {name} ({simple})')
                    lines.append(f'    {llm_code}')
                    lines.append('')
    
    lines.extend([
        f'    return {last_var}',
        '',
        '',
        'if __name__ == "__main__":',
        '    print("Pipeline loaded.")',
        "    print('Usage: run_pipeline(conn, df_input, SDtInicialMesAnt=\"2024-01-01\", SDtFinalMesAnt=\"2024-01-31\")')",
    ])
    
    return '\n'.join(lines)


def load_analysis_json(knwf_path, log):
    """Try to load pre-analyzed JSON. Returns (nodes, connections).
    
    Assigns a unique `_uid` (index-based) to each node so that
    duplicate `node_id` values from different metanodes are
    disambiguated throughout the pipeline.
    """
    paths_to_try = [
        knwf_path.parent / "workflow_analysis_v2.json",
        Path(__file__).parent / "workflow_analysis_v2.json",
    ]
    
    for path in paths_to_try:
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    nodes = data.get('nodes', [])
                    connections = data.get('connections', [])
                    # Assign unique _uid to each node (index-based)
                    for i, node in enumerate(nodes):
                        node['_uid'] = str(i)
                    log.extraction_method = f"JSON ({path.name})"
                    return nodes, connections
            except Exception as e:
                log.add_warning(f"Failed to load {path}: {e}")
    
    return None, []


def build_graph(nodes, connections, extract_dir=None):
    """Build port-aware graph using KNWF-native (path, node_id) keys.
    
    When extract_dir is provided (preferred), builds the graph from
    KNWF workflow.knime files where each node is uniquely identified
    by (full_parent_path, node_id). This eliminates all duplicate
    node_id ambiguity.
    
    Falls back to JSON-only connections when extract_dir is not available.
    
    Returns dict with:
        inputs: {_uid: {dest_port: {source: _uid, source_port: str}}}
        sorted_ids: topologically sorted _uid values
        id_to_index: {_uid: index_in_nodes_list}
        uid_to_node_id: {_uid: original_node_id}
    """
    from collections import defaultdict
    from xml.etree import ElementTree as ET
    import heapq
    
    # Phase 1: Map JSON _uid → index (guaranteed unique)
    id_to_index = {}
    for i, node in enumerate(nodes):
        uid = node.get('_uid', str(i))
        id_to_index[uid] = i
    
    # Phase 2: Build edges — use KNWF-native when possible
    inputs = defaultdict(dict)
    children = defaultdict(set)
    in_degree = defaultdict(int)
    all_uids = set(id_to_index.keys())
    
    if extract_dir:
        # === KNWF-NATIVE GRAPH (0-violation) ===
        extract_path = Path(extract_dir)
        NS = {'k': 'http://www.knime.org/2008/09/XMLConfig'}
        
        # 2a: Build (full_path, node_id) → sequential KNWF index
        knwf_registry = {}  # (parent_rel, node_id) → knwf_idx
        knwf_idx = 0
        for sf in sorted(extract_path.rglob('settings.xml')):
            m = re.search(r'#(\d+)\)', sf.parent.name)
            if not m:
                continue
            nid = m.group(1)
            try:
                gp_rel = str(sf.parent.parent.relative_to(extract_path))
            except ValueError:
                gp_rel = sf.parent.parent.name
            knwf_registry[(gp_rel, nid)] = knwf_idx
            knwf_idx += 1
        
        # 2b: Map KNWF indices to JSON _uid via node_id ordering
        # Group JSON _uids by node_id (in order)
        nid_to_json_uids = defaultdict(list)
        for node in nodes:
            nid = str(node.get('node_id', node.get('id', '')))
            nid_to_json_uids[nid].append(node.get('_uid', ''))
        
        # Group KNWF entries by node_id (in order)
        nid_to_knwf = defaultdict(list)
        for (gp_rel, nid), kidx in sorted(
                knwf_registry.items(), key=lambda x: x[1]):
            nid_to_knwf[nid].append((gp_rel, kidx))
        
        # Build knwf_key → JSON _uid mapping
        knwf_key_to_uid = {}  # (parent_rel, node_id) → _uid
        for nid in nid_to_json_uids:
            juids = nid_to_json_uids[nid]
            kentries = nid_to_knwf.get(nid, [])
            for (juid, (gp_rel, kidx)) in zip(juids, kentries):
                knwf_key_to_uid[(gp_rel, nid)] = juid
        
        # 2c: Extract connections from ALL workflow.knime files
        # using full path context for unambiguous resolution
        # Port-resolved edges are DEFERRED until after sort
        deferred_port_edges = []  # [(src_uid, dst_uid, dst_port, src_port)]
        
        for wf_file in sorted(extract_path.rglob('workflow.knime')):
            parent = wf_file.parent
            try:
                parent_rel = str(parent.relative_to(extract_path))
            except ValueError:
                parent_rel = parent.name
            
            m = re.search(r'#(\d+)\)', parent.name)
            meta_id = m.group(1) if m else None
            
            tree = ET.parse(wf_file)
            root = tree.getroot()
            
            port_in = {}   # port → [internal_dest_ids]
            port_out = {}  # port → [internal_src_ids]
            
            for cfg in root.findall(
                    './/k:config[@key="connections"]/k:config', NS):
                c = {}
                for e in cfg.findall('k:entry', NS):
                    c[e.get('key')] = e.get('value')
                
                src = c.get('sourceID', '')
                dst = c.get('destID', '')
                sp = c.get('sourcePort', '0')
                dp = c.get('destPort', '0')
                
                if src == '-1' and meta_id:
                    port_in.setdefault(sp, []).append(dst)
                elif dst == '-1' and meta_id:
                    port_out.setdefault(dp, []).append(src)
                elif src != '-1' and dst != '-1':
                    src_uid = knwf_key_to_uid.get((parent_rel, src))
                    dst_uid = knwf_key_to_uid.get((parent_rel, dst))
                    
                    if src_uid and dst_uid and src_uid != dst_uid:
                        inputs[dst_uid][dp] = {
                            'source': src_uid, 'source_port': sp}
                        children[src_uid].add(dst_uid)
                        in_degree[dst_uid] += 1
            
            # Collect metanode port edges (deferred)
            if meta_id and (port_in or port_out):
                gp = parent.parent
                try:
                    gp_rel = str(gp.relative_to(extract_path))
                except ValueError:
                    gp_rel = gp.name
                gp_wf = gp / 'workflow.knime'
                if gp_wf.exists():
                    gp_tree = ET.parse(gp_wf)
                    gp_root = gp_tree.getroot()
                    for cfg in gp_root.findall(
                            './/k:config[@key="connections"]'
                            '/k:config', NS):
                        gc = {}
                        for e in cfg.findall('k:entry', NS):
                            gc[e.get('key')] = e.get('value')
                        gs = gc.get('sourceID', '')
                        gd = gc.get('destID', '')
                        gsp = gc.get('sourcePort', '0')
                        gdp = gc.get('destPort', '0')
                        
                        if gd == meta_id and gs != '-1':
                            src_uid = knwf_key_to_uid.get(
                                (gp_rel, gs))
                            for idst in port_in.get(gdp, []):
                                dst_uid = knwf_key_to_uid.get(
                                    (parent_rel, idst))
                                if src_uid and dst_uid:
                                    deferred_port_edges.append(
                                        (src_uid, dst_uid,
                                         '1', gsp))
                        
                        if gs == meta_id and gd != '-1':
                            dst_uid = knwf_key_to_uid.get(
                                (gp_rel, gd))
                            for isrc in port_out.get(gsp, []):
                                src_uid = knwf_key_to_uid.get(
                                    (parent_rel, isrc))
                                if src_uid and dst_uid:
                                    deferred_port_edges.append(
                                        (src_uid, dst_uid,
                                         gdp, '1'))
    else:
        # === FALLBACK: JSON-only connections ===
        nid_to_uids = defaultdict(list)
        for node in nodes:
            nid = str(node.get('node_id', node.get('id', '')))
            nid_to_uids[nid].append(node.get('_uid', ''))
        
        for conn in connections:
            src_nid = str(conn.get('source_id', ''))
            dst_nid = str(conn.get('dest_id', ''))
            sp = str(conn.get('source_port', '0'))
            dp = str(conn.get('dest_port', '1'))
            
            src_cands = nid_to_uids.get(src_nid, [])
            dst_cands = nid_to_uids.get(dst_nid, [])
            src = src_cands[0] if src_cands else None
            dst = dst_cands[0] if dst_cands else None
            
            if src and dst and src != dst:
                inputs[dst][dp] = {'source': src, 'source_port': sp}
                children[src].add(dst)
                in_degree[dst] += 1
    
    # Phase 3: Topological sort (Kahn's with stable tie-breaking)
    heap = []
    for uid in all_uids:
        if in_degree[uid] == 0:
            heapq.heappush(heap, (id_to_index.get(uid, 999999), uid))
    
    sorted_ids = []
    while heap:
        _, uid = heapq.heappop(heap)
        sorted_ids.append(uid)
        for child in children.get(uid, set()):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                heapq.heappush(
                    heap, (id_to_index.get(child, 999999), child))
    
    # Add remaining nodes (cycles or disconnected)
    sorted_set = set(sorted_ids)
    for uid in all_uids:
        if uid not in sorted_set:
            sorted_ids.append(uid)
    
    # Phase 4: Add FORWARD-only port-resolved edges to inputs
    # Only add edges where source is sorted BEFORE destination,
    # ensuring get_upstream_var never references undefined variables
    if extract_dir and deferred_port_edges:
        pos = {uid: i for i, uid in enumerate(sorted_ids)}
        for src_uid, dst_uid, dp, sp in deferred_port_edges:
            if pos.get(src_uid, 999) < pos.get(dst_uid, 0):
                inputs[dst_uid][dp] = {
                    'source': src_uid, 'source_port': sp}
    
    return {
        'inputs': dict(inputs),
        'sorted_ids': sorted_ids,
        'id_to_index': id_to_index,

    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python transpile.py <arquivo.knwf>")
        print("")
        print("Output:")
        print("  - arquivo.py      (Python code)")
        print("  - arquivo_log.md  (diagnostic log)")
        sys.exit(1)
    
    knwf_path = Path(sys.argv[1]).resolve()
    
    if not knwf_path.exists():
        print(f"Error: File not found: {knwf_path}")
        sys.exit(1)
    
    output_path = knwf_path.with_suffix('.py')
    log_path = knwf_path.parent / f"{knwf_path.stem}_log.md"
    
    log = TranspilerLog(str(knwf_path))
    
    print("="*60)
    print("KNIME to Python Transpiler")
    print("="*60)
    print(f"Input:  {knwf_path}")
    print(f"Output: {output_path}")
    print(f"Log:    {log_path}")
    print("="*60)
    
    try:
        print("\n[1/5] Loading nodes...")
        
        # Always extract KNWF to get SQL queries
        log.extraction_method = "KNWF Extraction + SQL"
        extract_dir = extract_knwf(knwf_path, log)
        
        # First try to load from JSON for node ordering + connections
        json_nodes, connections = load_analysis_json(knwf_path, log)
        graph = None
        
        if json_nodes:
            print(f"      Found {len(json_nodes)} nodes (from JSON)")
            print(f"      Found {len(connections)} connections")
            # Enrich JSON nodes with all parameters from settings.xml
            nodes = enrich_nodes_with_params(json_nodes, extract_dir, log)
            
            # Build port-aware graph (Phase 2)
            if connections:
                print("[2/5] Building port-aware graph...")
                # KNWF-native graph uses (path, node_id) for 0-violation ordering
                graph = build_graph(nodes, connections, extract_dir=extract_dir)
                print(f"      Topological order: {len(graph['sorted_ids'])} nodes")
                print(f"      Wired inputs: {len(graph['inputs'])} nodes")
        else:
            # Extract nodes directly from settings.xml
            nodes = find_nodes_from_settings(extract_dir, log)
            print(f"      Found {len(nodes)} nodes (from extraction)")
        
        shutil.rmtree(extract_dir, ignore_errors=True)
        
        if not nodes:
            log.add_error("No nodes found in workflow")
            print("\n[!] No nodes found!")
        else:
            print("[3/5] Generating code...")
            code = generate_code(nodes, log, graph=graph)
            
            print("[4/5] Writing output...")
            try:
                output_path.write_text(code, encoding='utf-8')
                print(f"      Wrote {len(code)} bytes to {output_path.name}")
            except PermissionError as e:
                log.add_error(f"Permission denied writing to {output_path}: {e}")
                print(f"      ERROR: Permission denied - {output_path}")
            except OSError as e:
                log.add_error(f"OS error writing to {output_path}: {e}")
                print(f"      ERROR: OS error - {e}")
        
        print("[5/5] Writing log...")
        try:
            log_content = log.generate_markdown()
            log_path.write_text(log_content, encoding='utf-8')
            print(f"      Wrote {len(log_content)} bytes to {log_path.name}")
        except Exception as e:
            print(f"      ERROR writing log: {e}")
        
        matched = len(log.template_matches)
        fallback = len(log.fallback_nodes)
        total = len(log.nodes_found)
        coverage = (matched / total * 100) if total > 0 else 0
        
        print("\n" + "="*60)
        print("COMPLETE")
        print("="*60)
        print(f"Nodes:    {total}")
        print(f"Matched:  {matched}")
        print(f"Fallback: {fallback}")
        print(f"Coverage: {coverage:.1f}%")
        print("="*60)
        print(f"Output:   {output_path}")
        print(f"Log:      {log_path}")
        if log.errors:
            print("\nErrors:")
            for err in log.errors:
                print(f"  - {err}")
        print("="*60)
        
    except Exception as e:
        log.add_error(str(e))
        log_content = log.generate_markdown()
        log_path.write_text(log_content, encoding='utf-8')
        
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nLog written to: {log_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
