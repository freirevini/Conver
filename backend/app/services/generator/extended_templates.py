"""
Extended Node Templates

Additional KNIME node templates for Phase 3 expansion.
Adds 18 priority nodes + expression parser integration.
"""
from typing import Any, Dict, List, Optional, Tuple
from app.services.parsers import (
    math_parser,
    string_parser,
    rule_parser,
    column_expr_parser,
    get_parser_for_node
)

# ==================== Grupo A: Simple Template Nodes (15) ====================

EXTENDED_TEMPLATES: Dict[str, Dict[str, Any]] = {
    
    # -------------------- Table I/O --------------------
    "org.knime.base.node.io.table.read.TableReaderNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": "{output_var} = pd.read_pickle('{file_path}')",
        "description": "Read KNIME table file",
    },
    
    "org.knime.base.node.io.table.write.TableWriterNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": "{input_var}.to_pickle('{file_path}')",
        "description": "Write KNIME table file",
    },
    
    # -------------------- Column Splitting/Combining --------------------
    "org.knime.base.node.preproc.columnsplit.ColumnSplitterNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """{output_var}_included = {input_var}[{include_columns}]
{output_var}_excluded = {input_var}.drop(columns={include_columns})""",
        "description": "Split DataFrame by columns",
        "outputs": 2,
    },
    
    "org.knime.base.node.preproc.columncombiner.ColumnCombinerNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """{output_var} = {input_var}.copy()
{output_var}['{new_column}'] = {input_var}[{columns}].astype(str).agg('{separator}'.join, axis=1)""",
        "description": "Combine columns into one",
    },
    
    # -------------------- Row Splitting --------------------
    "org.knime.base.node.preproc.filter.row.splitter.RowSplitterNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """{output_var}_match = {input_var}[{condition}]
{output_var}_no_match = {input_var}[~({condition})]""",
        "description": "Split rows by condition",
        "outputs": 2,
    },
    
    # -------------------- Duplicate Handling --------------------
    "org.knime.base.node.preproc.duplicates.DuplicateRowFilterNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """{output_var} = {input_var}.drop_duplicates(subset={columns}, keep='{keep}')
{output_var}_duplicates = {input_var}[{input_var}.duplicated(subset={columns}, keep=False)]""",
        "description": "Filter duplicate rows",
        "outputs": 2,
    },
    
    # -------------------- Cell Operations --------------------
    "org.knime.base.node.preproc.cellsplit.CellSplitterNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """{output_var} = {input_var}.copy()
_split = {input_var}['{column}'].str.split('{delimiter}', expand=True)
_split.columns = ['{column}_' + str(i) for i in range(_split.shape[1])]
{output_var} = pd.concat([{output_var}, _split], axis=1)""",
        "description": "Split cell content into columns",
    },
    
    # -------------------- Pivot/Unpivot --------------------
    "org.knime.base.node.preproc.pivot.PivotNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": "{output_var} = {input_var}.pivot_table(index={group_columns}, columns='{pivot_column}', values='{value_column}', aggfunc='{aggfunc}').reset_index()",
        "description": "Pivot table",
    },
    
    "org.knime.base.node.preproc.unpivot.UnpivotNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": "{output_var} = {input_var}.melt(id_vars={id_columns}, value_vars={value_columns}, var_name='{var_name}', value_name='{value_name}')",
        "description": "Unpivot (melt) table",
    },
    
    # -------------------- Flow Control - Switches --------------------
    "org.knime.base.node.switches.emptytableswitch.EmptyTableSwitchNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# Empty Table Switch
if {input_var}.empty:
    {output_var} = {input_var}  # Route to empty branch
else:
    {output_var} = {input_var}  # Route to non-empty branch""",
        "description": "Switch based on empty table",
    },
    
    "org.knime.base.node.switches.ifswitch.IFSwitchNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# IF Switch - Branch based on condition
if {condition}:
    {output_var} = {input_var}  # Top port
else:
    {output_var} = {input_var}  # Bottom port""",
        "description": "IF Switch node",
    },
    
    "org.knime.base.node.switches.caseswitch.CaseSwitchNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# Case Switch - Multi-branch based on value
_case_value = {case_variable}
{output_var} = {input_var}  # Route based on case value""",
        "description": "Case Switch node",
    },
    
    # -------------------- Loops --------------------
    "org.knime.base.node.meta.looper.tablerow.TableRowToVariableLoopStartNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# Table Row to Variable Loop - Iterate over rows
for _row_idx, _row in {input_var}.iterrows():
    # Variables available: _row['column_name']
    {output_var} = pd.DataFrame([_row])""",
        "description": "Loop over table rows",
        "loop_type": "row_iterator",
    },
    
    "org.knime.base.node.meta.looper.columnlist.ColumnListLoopStartNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# Column List Loop - Iterate over columns
for _col_name in {input_var}[{columns}].columns:
    {output_var} = {input_var}[[_col_name]]""",
        "description": "Loop over columns",
        "loop_type": "column_iterator",
    },
    
    "org.knime.base.node.meta.looper.counting.CountingLoopStartNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# Counting Loop
for _iteration in range({start}, {end}, {step}):
    # Current iteration: _iteration
    {output_var} = {input_var}""",
        "description": "Counting loop",
        "loop_type": "counter",
    },
    
    "org.knime.base.node.meta.looper.LoopEndNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# Loop End - Collect results
{output_var} = pd.concat(_loop_results, ignore_index=True)""",
        "description": "End loop and collect",
    },
    
    # -------------------- End IF (Flow Control) --------------------
    "org.knime.base.node.switches.endcase.EndCaseNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# End IF - Merge branches back
# Takes first non-empty input from IF branches
{output_var} = {input_var}""",
        "description": "End IF merge node",
    },
    
    "org.knime.base.node.switches.endif.EndIfNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# End IF - Merge branches
{output_var} = {input_var}""",
        "description": "End IF node",
    },
    
    # -------------------- Group Loop Start/End --------------------
    "org.knime.base.node.meta.looper.group.GroupLoopStartNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# Group Loop Start - Iterate over groups
_groups = {input_var}.groupby({group_columns})
_loop_results = []
for _group_key, _group_df in _groups:
    {output_var} = _group_df.copy()""",
        "description": "Start loop over groups",
        "loop_type": "group_iterator",
    },
    
    "org.knime.base.node.meta.looper.variable.VariableLoopEndNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# Variable Loop End - Collect variable results
{output_var} = pd.DataFrame(_loop_results)""",
        "description": "End loop with variable collection",
    },
    
    # -------------------- Rule-based Row Filter --------------------
    "org.knime.base.node.rules.engine.filter.RuleBasedRowFilterNodeFactory": {
        "imports": ["import pandas as pd", "import numpy as np"],
        "code": """# Rule-based Row Filter
# Conditions defined by rules - filter matching rows
_mask = {filter_condition}
{output_var} = {input_var}[_mask]""",
        "description": "Filter rows by rule conditions",
    },
    
    "org.knime.base.node.rules.engine.splitter.RuleBasedRowSplitterNodeFactory": {
        "imports": ["import pandas as pd", "import numpy as np"],
        "code": """# Rule-based Row Splitter
_mask = {filter_condition}
{output_var}_match = {input_var}[_mask]
{output_var}_no_match = {input_var}[~_mask]""",
        "description": "Split rows by rule conditions",
        "outputs": 2,
    },
    
    # -------------------- Date/Time Operations --------------------
    "org.knime.time.node.calculate.DateTimeDifferenceNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# Date/Time Difference
{output_var} = {input_var}.copy()
{output_var}['{output_column}'] = ({input_var}['{end_column}'] - {input_var}['{start_column}']).dt.{granularity}""",
        "description": "Calculate difference between date/time columns",
    },
    
    "org.knime.time.node.create.createtimedatecell.CreateDateTimeNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# Create Date/Time Range
{output_var} = pd.DataFrame()
{output_var}['datetime'] = pd.date_range(start='{start}', end='{end}', freq='{freq}')""",
        "description": "Create date/time range",
    },
    
    "org.knime.time.node.manipulate.modifytime.ModifyTimeNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# Modify Time
{output_var} = {input_var}.copy()
{output_var}['{column}'] = {input_var}['{column}'] + pd.Timedelta({value}, unit='{unit}')""",
        "description": "Modify time values",
    },
    
    "org.knime.time.node.manipulate.modifydate.ModifyDateNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# Modify Date
{output_var} = {input_var}.copy()
{output_var}['{column}'] = {input_var}['{column}'] + pd.DateOffset({offset_params})""",
        "description": "Modify date values",
    },
    
    "org.knime.time.node.extract.datetime.ExtractDateTimeFieldsNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# Extract Date/Time Fields
{output_var} = {input_var}.copy()
{output_var}['year'] = {input_var}['{column}'].dt.year
{output_var}['month'] = {input_var}['{column}'].dt.month
{output_var}['day'] = {input_var}['{column}'].dt.day""",
        "description": "Extract date/time fields",
    },
    
    "org.knime.time.node.convert.durationtonumber.DurationToNumberNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# Duration to Number
{output_var} = {input_var}.copy()
{output_var}['{output_column}'] = {input_var}['{column}'].dt.total_seconds() / {divisor}""",
        "description": "Convert duration to number",
    },
    
    "org.knime.time.node.convert.legacytodate.LegacyToDateTimeNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# Legacy DateTime to DateTime
{output_var} = {input_var}.copy()
{output_var}['{column}'] = pd.to_datetime({input_var}['{column}'])""",
        "description": "Convert legacy date/time format",
    },
    
    # -------------------- Number/String Conversions --------------------
    "org.knime.base.node.preproc.numbertostring.NumberToStringNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# Number to String
{output_var} = {input_var}.copy()
{output_var}[{columns}] = {input_var}[{columns}].astype(str)""",
        "description": "Convert numbers to strings",
    },
    
    "org.knime.base.node.preproc.stringtonumber.StringToNumberNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# String to Number
{output_var} = {input_var}.copy()
{output_var}[{columns}] = pd.to_numeric({input_var}[{columns}], errors='coerce')""",
        "description": "Convert strings to numbers",
    },
    
    "org.knime.base.node.preproc.double2int.DoubleToIntNodeFactory": {
        "imports": ["import pandas as pd", "import numpy as np"],
        "code": """# Double to Int
{output_var} = {input_var}.copy()
{output_var}[{columns}] = {input_var}[{columns}].astype('Int64')""",
        "description": "Convert double to integer",
    },
    
    "org.knime.base.node.preproc.rounddouble.RoundDoubleNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# Round Double
{output_var} = {input_var}.copy()
{output_var}[{columns}] = {input_var}[{columns}].round({precision})""",
        "description": "Round double values",
    },
    
    # -------------------- Sorter --------------------
    "org.knime.base.node.preproc.sorter.SorterNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# Sorter
{output_var} = {input_var}.sort_values(by={columns}, ascending={ascending})""",
        "description": "Sort table by columns",
    },
    
    # -------------------- Database Legacy --------------------
    "org.knime.base.node.io.database.DBLooperNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# Database Looping (Legacy)
# Loop over query results
_query_results = []
for _value in {loop_values}:
    _query = '{query}'.format(value=_value)
    _result = pd.read_sql(_query, _connection)
    _query_results.append(_result)
{output_var} = pd.concat(_query_results, ignore_index=True)""",
        "description": "Database looping query",
    },
    
    "org.knime.base.node.io.database.DBReaderNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# Database Reader (Legacy)
{output_var} = pd.read_sql('{query}', _connection)""",
        "description": "Read from database",
    },
    
    "org.knime.base.node.io.database.DBConnectionNodeFactory": {
        "imports": ["import pandas as pd", "from sqlalchemy import create_engine"],
        "code": """# Database Connector (Legacy)
_connection = create_engine('{connection_string}')""",
        "description": "Create database connection",
    },
    
    # -------------------- Java Snippet (stub) --------------------
    "org.knime.base.node.jsnippet.JavaSnippetNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# Java Snippet - Manual translation required
# Original Java code: {java_code}
{output_var} = {input_var}.copy()
# TODO: Translate Java logic to Python""",
        "description": "Java Snippet (requires manual translation)",
    },
    
    # -------------------- Parameter Optimization Loop --------------------
    "org.knime.base.node.meta.looper.paramoptimizer.ParameterOptimizationLoopStartNodeFactory": {
        "imports": ["import pandas as pd", "import itertools"],
        "code": """# Parameter Optimization Loop Start
_param_grid = {param_grid}
_param_combinations = list(itertools.product(*_param_grid.values()))
_loop_results = []
for _params in _param_combinations:
    _current_params = dict(zip(_param_grid.keys(), _params))
    {output_var} = {input_var}""",
        "description": "Start parameter optimization loop",
    },
    
    "org.knime.base.node.meta.looper.paramoptimizer.ParameterOptimizationLoopEndNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# Parameter Optimization Loop End
{output_var} = pd.DataFrame(_loop_results)""",
        "description": "End parameter optimization loop",
    },
    
    # -------------------- Google BigQuery --------------------
    "org.knime.google.bigquery.connector.GoogleBigQueryConnectorNodeFactory": {
        "imports": ["import pandas as pd", "from google.cloud import bigquery"],
        "code": """# Google BigQuery Connector
_client = bigquery.Client(project='{project_id}')
{output_var} = _client.query('{query}').to_dataframe()""",
        "description": "Google BigQuery connection",
    },
    
    # -------------------- Final 7 nodes for 100% coverage --------------------
    "org.knime.time.node.manipulate.datetimeshift.DateTimeShiftNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# Date/Time Shift
{output_var} = {input_var}.copy()
{output_var}['{column}'] = {input_var}['{column}'] + pd.DateOffset({shift_params})""",
        "description": "Shift date/time values",
    },
    
    "org.knime.base.node.preproc.addemptyrows.AddEmptyRowsNodeFactory": {
        "imports": ["import pandas as pd", "import numpy as np"],
        "code": """# Add Empty Rows
_empty = pd.DataFrame(np.nan, index=range({num_rows}), columns={input_var}.columns)
{output_var} = pd.concat([{input_var}, _empty], ignore_index=True)""",
        "description": "Add empty rows to table",
    },
    
    "org.knime.base.node.io.database.DBQueryReaderNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# DB Query Reader
{output_var} = pd.read_sql('{query}', _connection)""",
        "description": "Execute SQL query and read results",
    },
    
    "org.knime.base.node.jsnippet.variable.JavaEditVariableNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# Java Edit Variable - Manual translation required
# Original Java code: {java_code}
{output_var} = {input_var}  # TODO: Translate Java variable logic""",
        "description": "Java variable editing (requires translation)",
    },
    
    "org.knime.python3.scripting.nodes.script.PythonScriptNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# Python Script
# Passthrough - Python script already native
{output_var} = {input_var}
# User script: {script}""",
        "description": "Python script passthrough",
    },
    
    "org.knime.base.node.preproc.counter.CounterNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# Counter Generation
{output_var} = {input_var}.copy()
{output_var}['counter'] = range(1, len({input_var}) + 1)""",
        "description": "Generate counter column",
    },
    
    "org.knime.time.node.convert.datetolegacy.DateTimeToLegacyNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# DateTime to Legacy DateTime
{output_var} = {input_var}.copy()
# Convert to legacy format (string representation)
{output_var}['{column}'] = {input_var}['{column}'].dt.strftime('%Y-%m-%d %H:%M:%S')""",
        "description": "Convert new DateTime to legacy format",
    },
    
    # -------------------- Google Authentication --------------------
    "org.knime.google.api.credential.GoogleAuthenticationApiKeyNodeFactory": {
        "imports": ["import pandas as pd"],
        "code": """# Google Authentication (API Key)
_api_key = '{api_key}'
# API key stored for subsequent Google API calls""",
        "description": "Google API Key authentication",
    },
}


# ==================== Grupo B: Database Connectors (4) ====================

DB_CONNECTOR_TEMPLATES: Dict[str, Dict[str, Any]] = {
    
    "org.knime.database.connector.MySQLConnectorNodeFactory": {
        "imports": [
            "import pandas as pd",
            "import mysql.connector",
        ],
        "code": """# MySQL Connection
_conn = mysql.connector.connect(
    host='{host}',
    port={port},
    user='{user}',
    password='{password}',
    database='{database}'
)
{output_var} = pd.read_sql('{query}', _conn)
_conn.close()""",
        "description": "MySQL database connection",
    },
    
    "org.knime.database.connector.PostgreSQLConnectorNodeFactory": {
        "imports": [
            "import pandas as pd",
            "import psycopg2",
        ],
        "code": """# PostgreSQL Connection
_conn = psycopg2.connect(
    host='{host}',
    port={port},
    user='{user}',
    password='{password}',
    dbname='{database}'
)
{output_var} = pd.read_sql('{query}', _conn)
_conn.close()""",
        "description": "PostgreSQL database connection",
    },
    
    "org.knime.database.connector.OracleConnectorNodeFactory": {
        "imports": [
            "import pandas as pd",
            "import cx_Oracle",
        ],
        "code": """# Oracle Connection
_dsn = cx_Oracle.makedsn('{host}', {port}, service_name='{service_name}')
_conn = cx_Oracle.connect(user='{user}', password='{password}', dsn=_dsn)
{output_var} = pd.read_sql('{query}', _conn)
_conn.close()""",
        "description": "Oracle database connection",
    },
    
    "org.knime.database.connector.H2ConnectorNodeFactory": {
        "imports": [
            "import pandas as pd",
            "import jaydebeapi",
        ],
        "code": """# H2 Database Connection
_conn = jaydebeapi.connect(
    'org.h2.Driver',
    '{jdbc_url}',
    ['{user}', '{password}'],
    '{driver_path}'
)
{output_var} = pd.read_sql('{query}', _conn)
_conn.close()""",
        "description": "H2 database connection",
    },
}

# ==================== Expression Node Factory Classes ====================

EXPRESSION_NODES = {
    "org.knime.base.node.preproc.stringmanipulation.StringManipulationNodeFactory": "string_parser",
    "org.knime.base.node.preproc.mathformula.MathFormulaNodeFactory": "math_parser",
    "org.knime.base.node.rules.engine.RuleEngineNodeFactory": "rule_parser",
    "org.knime.base.node.rules.engine.RuleEngineNodeFactory2": "rule_parser",
    "org.knime.base.expressions.node.ExpressionNodeFactory": "column_expr_parser",
    "org.knime.base.node.preproc.colexpressions.ColumnExpressionsNodeFactory": "column_expr_parser",
}


def get_all_extended_templates() -> Dict[str, Dict[str, Any]]:
    """Get all extended templates (Grupo A + Grupo B)."""
    all_templates = {}
    all_templates.update(EXTENDED_TEMPLATES)
    all_templates.update(DB_CONNECTOR_TEMPLATES)
    return all_templates


def generate_expression_code(
    factory_class: str,
    expression: str,
    output_column: str,
    input_var: str = "df"
) -> Tuple[str, List[str]]:
    """
    Generate Python code for expression nodes using parsers.
    
    Args:
        factory_class: KNIME factory class
        expression: KNIME expression string
        output_column: Name for output column
        input_var: Input DataFrame variable name
        
    Returns:
        Tuple of (code, imports)
    """
    parser_name = EXPRESSION_NODES.get(factory_class)
    
    if parser_name == "string_parser":
        parser = string_parser
        parser.df_var = input_var
        code = parser.parse(expression, output_column)
        return code, parser.imports
    
    elif parser_name == "math_parser":
        parser = math_parser
        parser.df_var = input_var
        code = parser.parse(expression, output_column)
        return code, parser.imports
    
    elif parser_name == "rule_parser":
        parser = rule_parser
        parser.df_var = input_var
        code = parser.parse(expression, output_column)
        return code, parser.imports
    
    elif parser_name == "column_expr_parser":
        parser = column_expr_parser
        parser.df_var = input_var
        code = parser.parse(expression, output_column)
        return code, parser.imports
    
    return f"# Unknown expression node: {factory_class}", ["import pandas as pd"]


def is_expression_node(factory_class: str) -> bool:
    """Check if a node requires expression parsing."""
    return factory_class in EXPRESSION_NODES


# Count stats
def get_template_stats() -> Dict[str, int]:
    """Get statistics about available templates."""
    return {
        "grupo_a_nodes": len(EXTENDED_TEMPLATES),
        "grupo_b_db_connectors": len(DB_CONNECTOR_TEMPLATES),
        "expression_nodes": len(EXPRESSION_NODES),
        "total_new": len(EXTENDED_TEMPLATES) + len(DB_CONNECTOR_TEMPLATES),
    }
