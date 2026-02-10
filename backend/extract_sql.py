#!/usr/bin/env python
"""
SQL Query Extractor for KNIME Nodes

Analyzes KNIME workflow settings.xml files to extract SQL queries
from Database Reader, DB Query Reader, and Database Looping nodes.
"""
import sys
import zipfile
import tempfile
import shutil
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional


def extract_knwf(knwf_path: Path) -> Path:
    """Extract KNWF to temp directory."""
    temp_dir = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(knwf_path, 'r') as zf:
        zf.extractall(temp_dir)
    return temp_dir


def find_sql_nodes(extract_dir: Path) -> List[Dict[str, Any]]:
    """Find all SQL-related nodes and extract their queries."""
    sql_patterns = [
        'DBReaderNodeFactory',
        'DBQueryReaderNodeFactory', 
        'DatabaseLoopingNodeFactory',
        'DBLoaderNodeFactory',
        'BigQueryDBConnectorNodeFactory',
    ]
    
    nodes = []
    
    for settings_file in extract_dir.rglob('settings.xml'):
        try:
            content = settings_file.read_text(encoding='utf-8', errors='ignore')
            
            # Check if this is a SQL-related node
            is_sql_node = any(pattern in content for pattern in sql_patterns)
            
            if not is_sql_node:
                continue
            
            node_name = settings_file.parent.name
            node_info = {
                'name': node_name,
                'path': str(settings_file.parent.relative_to(extract_dir)),
                'factory': '',
                'sql_query': '',
                'connection_info': {},
            }
            
            # Find factory
            for pattern in sql_patterns:
                if pattern in content:
                    node_info['factory'] = pattern
                    break
            
            # Extract SQL statement using multiple patterns
            sql_extracted = extract_sql_from_content(content)
            if sql_extracted:
                node_info['sql_query'] = sql_extracted
            
            # Extract connection info
            connection_info = extract_connection_info(content)
            node_info['connection_info'] = connection_info
            
            nodes.append(node_info)
            
        except Exception as e:
            print(f"Error reading {settings_file}: {e}")
    
    return nodes


def extract_sql_from_content(content: str) -> str:
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
    
    # Pattern 3: SQL in CDATA or multiline
    if not sql_query:
        # Look for SELECT...FROM pattern
        match = re.search(r'SELECT\s+.+?\s+FROM\s+.+?(?:WHERE|ORDER|GROUP|LIMIT|;|$)', content, 
                         re.IGNORECASE | re.DOTALL)
        if match:
            sql_query = match.group(0)
    
    # Pattern 4: value attribute containing SELECT
    if not sql_query:
        matches = re.findall(r'value="([^"]*SELECT[^"]*)"', content, re.IGNORECASE)
        if matches:
            # Get the longest match (likely the full query)
            sql_query = max(matches, key=len)
    
    # Decode KNIME escape sequences first
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


def decode_knime_escapes(text: str) -> str:
    """Decode KNIME-specific escape sequences like %%00010 (newline), %%00009 (tab)."""
    # %%XXXXX format - 5 digit ASCII code
    def replace_knime_escape(match):
        code = int(match.group(1))
        try:
            return chr(code)
        except ValueError:
            return match.group(0)
    
    # Replace %%00010 -> \n, %%00009 -> \t, etc.
    decoded = re.sub(r'%%(\d{5})', replace_knime_escape, text)
    
    return decoded


def extract_connection_info(content: str) -> Dict[str, str]:
    """Extract database connection information."""
    info = {}
    
    # Database URL
    match = re.search(r'<entry\s+key="database"\s+[^>]*value="([^"]*)"', content)
    if match:
        info['database'] = match.group(1)
    
    # JDBC URL
    match = re.search(r'<entry\s+key="databaseUrl"\s+[^>]*value="([^"]*)"', content)
    if match:
        info['jdbc_url'] = match.group(1)
    
    # Driver
    match = re.search(r'<entry\s+key="driver"\s+[^>]*value="([^"]*)"', content)
    if match:
        info['driver'] = match.group(1)
    
    return info


def generate_python_code(nodes: List[Dict[str, Any]]) -> str:
    """Generate Python code for SQL queries."""
    lines = [
        '"""',
        'Auto-generated SQL queries from KNIME workflow',
        '"""',
        'import pandas as pd',
        'from sqlalchemy import create_engine',
        '',
        '',
        '# Database connection (configure as needed)',
        '# engine = create_engine("mysql+pymysql://user:pass@host/db")',
        '',
    ]
    
    for i, node in enumerate(nodes):
        name = node['name']
        sql = node['sql_query']
        factory = node['factory']
        
        lines.append(f'# Node: {name}')
        lines.append(f'# Factory: {factory}')
        
        if node['connection_info']:
            for key, val in node['connection_info'].items():
                lines.append(f'# {key}: {val}')
        
        lines.append('')
        
        if sql:
            # Create function for this query
            func_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)[:40]
            lines.append(f'def query_{i}_{func_name}(conn):')
            lines.append('    """')
            lines.append(f'    {name}')
            lines.append('    """')
            lines.append('    sql = """')
            
            # Format SQL nicely
            sql_formatted = format_sql(sql)
            for sql_line in sql_formatted.split('\n'):
                lines.append(f'        {sql_line}')
            
            lines.append('    """')
            lines.append('    return pd.read_sql(sql, conn)')
            lines.append('')
        else:
            lines.append(f'# WARNING: No SQL query found in {name}')
            lines.append('')
    
    return '\n'.join(lines)


def format_sql(sql: str) -> str:
    """Basic SQL formatting."""
    # Add newlines before major clauses
    keywords = ['SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'JOIN', 'LEFT JOIN', 
                'RIGHT JOIN', 'INNER JOIN', 'ON', 'GROUP BY', 'ORDER BY', 
                'HAVING', 'LIMIT', 'UNION']
    
    formatted = sql
    for kw in keywords:
        formatted = re.sub(rf'\s+({kw})\s+', rf'\n{kw} ', formatted, flags=re.IGNORECASE)
    
    return formatted.strip()


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_sql.py <arquivo.knwf>")
        sys.exit(1)
    
    knwf_path = Path(sys.argv[1]).resolve()
    
    if not knwf_path.exists():
        print(f"Error: File not found: {knwf_path}")
        sys.exit(1)
    
    print("="*70)
    print("KNIME SQL Query Extractor")
    print("="*70)
    print(f"Input: {knwf_path}")
    print("="*70)
    
    # Extract
    print("\n[1/3] Extracting workflow...")
    extract_dir = extract_knwf(knwf_path)
    
    # Find SQL nodes
    print("[2/3] Analyzing SQL nodes...")
    nodes = find_sql_nodes(extract_dir)
    print(f"      Found {len(nodes)} SQL nodes")
    
    # Display results
    print("\n" + "="*70)
    print("SQL NODES FOUND")
    print("="*70)
    
    for node in nodes:
        print(f"\nNode: {node['name']}")
        print(f"Factory: {node['factory']}")
        print(f"Path: {node['path']}")
        
        if node['connection_info']:
            print("Connection:")
            for k, v in node['connection_info'].items():
                print(f"  {k}: {v}")
        
        if node['sql_query']:
            print("SQL Query:")
            print("-" * 40)
            print(node['sql_query'][:500])
            if len(node['sql_query']) > 500:
                print(f"... ({len(node['sql_query'])} chars total)")
            print("-" * 40)
        else:
            print("SQL Query: NOT FOUND")
    
    # Generate Python code
    print("\n[3/3] Generating Python code...")
    output_path = knwf_path.parent / f"{knwf_path.stem}_sql_queries.py"
    code = generate_python_code(nodes)
    output_path.write_text(code, encoding='utf-8')
    print(f"      Wrote to: {output_path}")
    
    # Cleanup
    shutil.rmtree(extract_dir, ignore_errors=True)
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
