"""
KNIME Node Template Mapper

Maps KNIME factory classes to Python code templates.
Provides deterministic code generation for common nodes (80% of cases).
"""
import logging
from typing import Dict, Optional, List, Tuple, Any

logger = logging.getLogger(__name__)

# Will import extended templates lazily to avoid circular imports
_extended_templates = None


class TemplateMapper:
    """
    Maps KNIME node factory classes to Python/Pandas code templates.
    
    Template structure:
    {
        'imports': List[str],  # Required imports
        'code': str,           # Python code template with placeholders
        'description': str     # What the node does
    }
    
    Placeholders:
    - {input_var}: Input DataFrame variable name
    - {output_var}: Output DataFrame variable name
    - {columns}: Column list from configuration
    - {condition}: Filter condition
    - {file_path}: File path for readers/writers
    - {settings}: Node-specific settings dict
    """
    
    # Common KNIME factory class patterns -> template mappings
    TEMPLATES: Dict[str, Dict[str, Any]] = {
        # ==================== I/O Nodes ====================
        "org.knime.base.node.io.csvreader.CSVReaderNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = pd.read_csv('{file_path}', sep='{separator}', header={header})",
            "description": "Read CSV file into DataFrame",
            "extract_settings": ["file_path", "separator", "header"]
        },
        
        "org.knime.base.node.io.csvwriter.CSVWriterNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{input_var}.to_csv('{file_path}', index=False, sep='{separator}')",
            "description": "Write DataFrame to CSV file",
            "extract_settings": ["file_path", "separator"]
        },
        
        "org.knime.ext.poi3.node.io.filehandling.excel.reader.ExcelTableReaderNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = pd.read_excel('{file_path}', sheet_name={sheet})",
            "description": "Read Excel file into DataFrame"
        },
        
        "org.knime.ext.poi3.node.io.filehandling.excel.writer.ExcelTableWriterNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{input_var}.to_excel('{file_path}', index=False, sheet_name='{sheet}')",
            "description": "Write DataFrame to Excel file"
        },
        
        # ==================== Column Operations ====================
        "org.knime.base.node.preproc.filter.column.DataColumnSpecFilterNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = {input_var}[{columns}]",
            "description": "Filter columns (keep selected)"
        },
        
        "org.knime.base.node.preproc.column.renamer.ColumnRenamerNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = {input_var}.rename(columns={rename_mapping})",
            "description": "Rename columns"
        },
        
        "org.knime.base.node.preproc.columtrans.ColumnResorterNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = {input_var}[{column_order}]",
            "description": "Reorder columns"
        },
        
        # ==================== Row Operations ====================
        "org.knime.base.node.preproc.filter.row.RowFilterNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = {input_var}[{condition}]",
            "description": "Filter rows by condition"
        },
        
        "org.knime.base.node.preproc.filter.row2.RowFilterNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = {input_var}.query('{condition}')",
            "description": "Filter rows by condition (v2)"
        },
        
        "org.knime.base.node.preproc.sample.RowSamplingNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = {input_var}.sample(n={n_samples}, random_state=42)",
            "description": "Sample N rows from DataFrame"
        },
        
        "org.knime.base.node.preproc.sorter.SorterNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = {input_var}.sort_values(by={sort_columns}, ascending={ascending})",
            "description": "Sort by columns"
        },
        
        # ==================== Aggregation ====================
        "org.knime.base.node.preproc.groupby.GroupByNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = {input_var}.groupby({group_columns}).agg({agg_dict}).reset_index()",
            "description": "Group by columns and aggregate"
        },
        
        "org.knime.base.node.preproc.columnTrans.aggregate.AggregateNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = {input_var}.agg({agg_dict})",
            "description": "Aggregate columns"
        },
        
        # Column Aggregator (concatenate values in columns)
        "org.knime.base.node.preproc.columnaggregator.ColumnAggregatorNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = {input_var}.groupby({group_columns}).agg({agg_dict}).reset_index()",
            "description": "Column Aggregator"
        },
        
        # Column Rename
        "org.knime.base.node.preproc.rename.RenameNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = {input_var}.rename(columns={rename_mapping})",
            "description": "Rename columns"
        },
        
        # Constant Value Column
        "org.knime.base.node.preproc.constantvalue.ConstantValueColumnNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = {input_var}.copy()\n{output_var}['{new_column}'] = {constant_value}",
            "description": "Add constant value column"
        },
        
        # Variable To Table
        "org.knime.base.node.flowvariable.variabletotablerow4.VariableToTable4NodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = pd.DataFrame([{flow_variables}])",
            "description": "Convert flow variables to table row"
        },
        
        # Table Row To Variable  
        "org.knime.base.node.flowvariable.tablerowtoVariable.TableRowToVariableNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "# Extract first row as variables\nflow_vars = {input_var}.iloc[0].to_dict() if not {input_var}.empty else {{}}\n{output_var} = {input_var}",
            "description": "Convert table row to flow variables"
        },
        
        # ==================== Joining ====================
        "org.knime.base.node.preproc.joiner.JoinerNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = pd.merge({left_var}, {right_var}, on={join_columns}, how='{join_type}')",
            "description": "Join two DataFrames"
        },
        
        "org.knime.base.node.preproc.joiner2.Joiner2NodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = pd.merge({left_var}, {right_var}, left_on={left_columns}, right_on={right_columns}, how='{join_type}')",
            "description": "Join two DataFrames (v2)"
        },
        
        "org.knime.base.node.preproc.crossjoiner.CrossJoinerNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = {left_var}.merge({right_var}, how='cross')",
            "description": "Cross join (Cartesian product)"
        },
        
        # ==================== Concatenation ====================
        "org.knime.base.node.preproc.append.row.AppendedRowsNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = pd.concat([{input_vars}], ignore_index=True)",
            "description": "Concatenate DataFrames vertically"
        },
        
        "org.knime.base.node.preproc.append.col.AppendedColumnsNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = pd.concat([{input_vars}], axis=1)",
            "description": "Concatenate DataFrames horizontally"
        },
        
        # ==================== String Manipulation ====================
        "org.knime.base.node.preproc.stringmanipulation.StringManipulationNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = {input_var}.copy()\n{output_var}['{new_column}'] = {expression}",
            "description": "String manipulation"
        },
        
        "org.knime.base.node.preproc.cellreplace.CellReplacerNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = {input_var}.copy()\n{output_var}['{column}'] = {input_var}['{column}'].replace({replace_dict})",
            "description": "Replace cell values"
        },
        
        # ==================== Math Operations ====================
        "org.knime.base.node.preproc.mathformula.MathFormulaNodeFactory": {
            "imports": ["import pandas as pd", "import numpy as np"],
            "code": "{output_var} = {input_var}.copy()\n{output_var}['{new_column}'] = {expression}",
            "description": "Mathematical formula"
        },
        
        # JEP Math Formula (ext.jep package)
        "org.knime.ext.jep.JEPNodeFactory": {
            "imports": ["import pandas as pd", "import numpy as np"],
            "code": "{output_var} = {input_var}.copy()\n{output_var}['{new_column}'] = {expression}",
            "description": "JEP Math Formula"
        },
        
        # ==================== Missing Values ====================
        "org.knime.base.node.preproc.pmml.missingval.MissingValueHandlerNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = {input_var}.fillna({fill_value})",
            "description": "Handle missing values"
        },
        
        # ==================== Date/Time Operations ====================
        "org.knime.time.node.calculate.datetimedifference.DateTimeDifferenceNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = {input_var}.copy()\n{output_var}['{new_column}'] = ({input_var}['{end_column}'] - {input_var}['{start_column}']).dt.total_seconds() / 86400",
            "description": "Calculate DateTime difference"
        },
        
        "org.knime.time.node.convert.oldtonew.OldToNewTimeNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = {input_var}.copy()\nfor col in {date_columns}:\n    {output_var}[col] = pd.to_datetime({output_var}[col])",
            "description": "Convert legacy date to new DateTime"
        },
        
        "org.knime.time.node.convert.newtoold.NewToOldTimeNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = {input_var}.copy()\nfor col in {date_columns}:\n    {output_var}[col] = {output_var}[col].dt.strftime('%Y-%m-%d %H:%M:%S')",
            "description": "Convert new DateTime to legacy format"
        },
        
        "org.knime.time.node.create.createdatetime.CreateDateTimeNodeFactory": {
            "imports": ["import pandas as pd", "from datetime import datetime"],
            "code": "{output_var} = {input_var}.copy()\n{output_var}['{new_column}'] = pd.Timestamp.now()",
            "description": "Create DateTime column"
        },
        
        "org.knime.time.node.extract.datetime.ExtractDateTimeFieldsNodeFactory2": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = {input_var}.copy()\n{output_var}['year'] = {input_var}['{date_column}'].dt.year\n{output_var}['month'] = {input_var}['{date_column}'].dt.month\n{output_var}['day'] = {input_var}['{date_column}'].dt.day",
            "description": "Extract DateTime fields"
        },
        
        # ==================== Database Connectors ====================
        "org.knime.base.node.io.database.connection.JDBCConnectorNodeFactory": {
            "imports": ["import sqlalchemy", "import os"],
            "code": "# Database connection\n_db_url = f\"{{db_type}}://{{db_user}}:{{db_password}}@{{db_host}}:{{db_port}}/{{db_name}}\"\n_engine = sqlalchemy.create_engine(_db_url)\n{output_var} = _engine",
            "description": "JDBC Database connection"
        },
        
        "org.knime.base.node.io.database.DatabaseLoopingNodeFactory": {
            "imports": ["import pandas as pd", "import sqlalchemy"],
            "code": "# Database looping query\n{output_var} = pd.read_sql('{query}', _engine)",
            "description": "Database looping query"
        },
        
        "org.knime.google.api.nodes.connector.GoogleApiConnectorFactory": {
            "imports": ["import os"],
            "code": "# Google API connector\n_google_credentials = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')\n{output_var} = {{'credentials': _google_credentials}}",
            "description": "Google API connector"
        },
        
        "org.knime.database.extension.bigquery.node.connector.BigQueryDBConnectorNodeFactory": {
            "imports": ["import os"],
            "code": "# BigQuery connector\n_bq_project = os.environ.get('GCP_PROJECT_ID', '{project_id}')\n{output_var} = {{'project': _bq_project}}",
            "description": "BigQuery database connector"
        },
        
        # ==================== Utility Nodes ====================
        "org.knime.datageneration.counter.CounterGenerationNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = {input_var}.copy()\n{output_var}['counter'] = range(1, len({input_var}) + 1)",
            "description": "Generate counter column"
        },
        
        "org.knime.base.node.jsnippet.JavaEditVarNodeFactory": {
            "imports": [],
            "code": "# Java Edit Variable - flow variable manipulation\n# Variables are already available in the Python scope\n{output_var} = {input_var}",
            "description": "Java Edit Variable (flow variable)"
        },
        
        "org.knime.base.node.preproc.colMissValue.ColumnMissingValueHandlerNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = {input_var}.copy()\nfor col in {columns}:\n    {output_var}[col] = {output_var}[col].fillna(method='{method}')",
            "description": "Handle missing values by column"
        },
        
        # ==================== Date/Time ====================
        "org.knime.time.node.create.createdate.CreateDateTimeNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = {input_var}.copy()\n{output_var}['{column}'] = pd.to_datetime({expression})",
            "description": "Create date/time column"
        },
        
        "org.knime.time.node.calculate.datetimediff.DateTimeDifferenceNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = {input_var}.copy()\n{output_var}['{new_column}'] = ({input_var}['{col1}'] - {input_var}['{col2}']).dt.{unit}",
            "description": "Calculate date/time difference"
        },
        
        # ==================== ML - Preprocessing ====================
        "org.knime.base.node.preproc.normalize.NormalizeNodeFactory": {
            "imports": [
                "import pandas as pd",
                "from sklearn.preprocessing import StandardScaler"
            ],
            "code": """scaler = StandardScaler()
{output_var} = {input_var}.copy()
{output_var}[{columns}] = scaler.fit_transform({input_var}[{columns}])""",
            "description": "Normalize numeric columns"
        },
        
        # ==================== ML - Classification ====================
        "org.knime.base.node.mine.decisiontree2.learner2.DecTreeLearnerNodeFactory2": {
            "imports": [
                "import pandas as pd",
                "from sklearn.tree import DecisionTreeClassifier"
            ],
            "code": """dt_model = DecisionTreeClassifier(max_depth={max_depth}, random_state=42)
dt_model.fit({input_var}[{feature_columns}], {input_var}['{target_column}'])""",
            "description": "Train Decision Tree classifier"
        },
        
        "org.knime.base.node.mine.bayes.naivebayes.learner2.NaiveBayesLearnerNodeFactory2": {
            "imports": [
                "import pandas as pd",
                "from sklearn.naive_bayes import GaussianNB"
            ],
            "code": """nb_model = GaussianNB()
nb_model.fit({input_var}[{feature_columns}], {input_var}['{target_column}'])""",
            "description": "Train Naive Bayes classifier"
        },
        
        # ==================== ML - Clustering ====================
        "org.knime.base.node.mine.cluster.kmeans.KMeansNodeFactory": {
            "imports": [
                "import pandas as pd",
                "from sklearn.cluster import KMeans"
            ],
            "code": """kmeans = KMeans(n_clusters={n_clusters}, random_state=42)
{output_var} = {input_var}.copy()
{output_var}['cluster'] = kmeans.fit_predict({input_var}[{feature_columns}])""",
            "description": "K-Means clustering"
        },
        
        # ==================== Flow Control ====================
        "org.knime.base.node.switches.endcase.EndcaseNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "# End Case node - flow control\n{output_var} = {input_var}",
            "description": "End case/switch node"
        },
        
        "org.knime.base.node.switches.startcase.StartcaseNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "# Start Case node - branch selection\n{output_var} = {input_var}",
            "description": "Start case/switch node"
        },
        
        # ==================== Rule Engine ====================
        "org.knime.base.node.rules.engine.RuleEngineNodeFactory": {
            "imports": ["import pandas as pd", "import numpy as np"],
            "code": """{output_var} = {input_var}.copy()
# Apply rule-based logic
conditions = {conditions}
{output_var}['{output_column}'] = np.select(conditions, {values}, default={default})""",
            "description": "Apply rule-based logic"
        },
        
        # ==================== Table Operations ====================
        "org.knime.base.node.preproc.table.rowtovar.TableRowToVariableNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "# Extract first row as variables\n{variables} = {input_var}.iloc[0].to_dict()",
            "description": "Convert row to variables"
        },
        
        "org.knime.base.node.flowvariable.constantvalue.ConstantValueNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": "{output_var} = {input_var}.copy()\n{output_var}['{column}'] = {value}",
            "description": "Add constant value column"
        },
        
        # ==================== Conversion Nodes (Task 1.2) ====================
        # Number to String conversion
        "org.knime.base.node.preproc.pmml.numbertostring.NumberToStringNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": """{output_var} = {input_var}.copy()
for col in {columns}:
    {output_var}[col] = {output_var}[col].astype(str)""",
            "description": "Convert numeric columns to string",
            "extract_settings": ["columns"]
        },
        
        "org.knime.base.node.preproc.pmml.numbertostring3.NumberToString3NodeFactory": {
            "imports": ["import pandas as pd"],
            "code": """{output_var} = {input_var}.copy()
for col in {columns}:
    {output_var}[col] = {output_var}[col].astype(str)""",
            "description": "Convert numeric columns to string (v3)",
            "extract_settings": ["columns"]
        },
        
        # String to Number conversion
        "org.knime.base.node.preproc.pmml.stringtonumber.StringToNumberNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": """{output_var} = {input_var}.copy()
for col in {columns}:
    {output_var}[col] = pd.to_numeric({output_var}[col], errors='coerce')""",
            "description": "Convert string columns to numeric",
            "extract_settings": ["columns"]
        },
        
        # Round Double
        "org.knime.base.node.preproc.rounddouble.RoundDoubleNodeFactory": {
            "imports": ["import pandas as pd", "import numpy as np"],
            "code": """{output_var} = {input_var}.copy()
for col in {columns}:
    {output_var}[col] = {output_var}[col].round({precision})""",
            "description": "Round numeric columns to specified precision",
            "extract_settings": ["columns", "precision"]
        },
        
        "org.knime.base.node.preproc.rounddouble.RoundDouble2NodeFactory": {
            "imports": ["import pandas as pd", "import numpy as np"],
            "code": """{output_var} = {input_var}.copy()
for col in {columns}:
    {output_var}[col] = {output_var}[col].round({precision})""",
            "description": "Round numeric columns (v2)",
            "extract_settings": ["columns", "precision"]
        },
        
        # Date/Time to String
        "org.knime.time.node.convert.datetimetostring.DateTimeToStringNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": """{output_var} = {input_var}.copy()
for col in {columns}:
    {output_var}[col] = {output_var}[col].dt.strftime('{format}')""",
            "description": "Convert datetime columns to string with format",
            "extract_settings": ["columns", "format"]
        },
        
        # String to Date/Time
        "org.knime.time.node.convert.stringtodatetime.StringToDateTimeNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": """{output_var} = {input_var}.copy()
for col in {columns}:
    {output_var}[col] = pd.to_datetime({output_var}[col], format='{format}', errors='coerce')""",
            "description": "Convert string columns to datetime",
            "extract_settings": ["columns", "format"]
        },
        
        # Double to Int
        "org.knime.base.node.preproc.doubletoint.DoubleToIntNodeFactory": {
            "imports": ["import pandas as pd", "import numpy as np"],
            "code": """{output_var} = {input_var}.copy()
for col in {columns}:
    {output_var}[col] = {output_var}[col].fillna(0).astype(int)""",
            "description": "Convert double columns to integer",
            "extract_settings": ["columns"]
        },
        
        # Int to Double
        "org.knime.base.node.preproc.inttodouble.IntToDoubleNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": """{output_var} = {input_var}.copy()
for col in {columns}:
    {output_var}[col] = {output_var}[col].astype(float)""",
            "description": "Convert integer columns to double",
            "extract_settings": ["columns"]
        },
        
        # Column Type Changer (generic type conversion)
        "org.knime.base.node.preproc.columnTypeChange.ColumnTypeChangerNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": """{output_var} = {input_var}.copy()
# Type mapping: {type_mapping}
for col, dtype in {type_mapping}.items():
    if dtype == 'string':
        {output_var}[col] = {output_var}[col].astype(str)
    elif dtype == 'int':
        {output_var}[col] = pd.to_numeric({output_var}[col], errors='coerce').fillna(0).astype(int)
    elif dtype == 'double':
        {output_var}[col] = pd.to_numeric({output_var}[col], errors='coerce')
    elif dtype == 'datetime':
        {output_var}[col] = pd.to_datetime({output_var}[col], errors='coerce')""",
            "description": "Change column types",
            "extract_settings": ["type_mapping"]
        },
        
        # ==================== Variables Nodes (Task 1.3) ====================
        # Table Row to Variable (standard)
        "org.knime.base.node.preproc.table.rowtovar.TableRowToVariableNodeFactory2": {
            "imports": ["import pandas as pd"],
            "code": """{output_var} = {input_var}.copy()
# Extract first row values as flow variables
if not {input_var}.empty:
    _flow_vars = {input_var}.iloc[0].to_dict()
    for key, value in _flow_vars.items():
        locals()[f'v_{{key}}'] = value""",
            "description": "Convert first row to flow variables"
        },
        
        # Table Row to Variable Loop Start
        "org.knime.base.node.flowvariable.tablerowtovariable.TableRowToVariableNodeFactory3": {
            "imports": ["import pandas as pd"],
            "code": """# Table Row to Variable Loop - iterate over rows
for _idx, _row in {input_var}.iterrows():
    {output_var} = pd.DataFrame([_row])
    # Loop body with row variables available
    _row_vars = _row.to_dict()""",
            "description": "Loop over rows converting each to variables",
            "is_loop_start": True
        },
        
        # Variable to Table Row
        "org.knime.base.node.flowvariable.variabletotablerow.VariableToTableRowNodeFactory2": {
            "imports": ["import pandas as pd"],
            "code": """# Convert flow variables to table row
{output_var} = pd.DataFrame([{variables}])""",
            "description": "Convert flow variables to single-row table",
            "extract_settings": ["variables"]
        },
        
        "org.knime.base.node.flowvariable.variabletotablerow3.VariableToTableRow3NodeFactory": {
            "imports": ["import pandas as pd"],
            "code": """# Convert flow variables to table row (v3)
{output_var} = pd.DataFrame([{variables}])""",
            "description": "Convert flow variables to single-row table (v3)",
            "extract_settings": ["variables"]
        },
        
        # Java Edit Variable (String Manipulation)
        "org.knime.base.node.flowvariable.stringmanipulation.edit.StringManipulationEditVariableNodeFactory": {
            "imports": ["import pandas as pd", "import re"],
            "code": """# Java Edit Variable - String manipulation on flow variables
{output_var} = {input_var}.copy()
# Expression: {expression}
{variable_name} = {expression_result}""",
            "description": "Edit flow variables using string manipulation",
            "extract_settings": ["expression", "variable_name"]
        },
        
        # Java Snippet Variable
        "org.knime.ext.javasnippet.node.JavaSnippetNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": """# Java Snippet converted to Python
{output_var} = {input_var}.copy()
# Original Java code: {java_code}
{output_var}['{output_column}'] = {output_var}.apply(lambda row: {python_expression}, axis=1)""",
            "description": "Custom Java snippet converted to Python",
            "extract_settings": ["java_code", "output_column"]
        },
        
        # String Manipulation (Variable)
        "org.knime.base.node.preproc.stringmanipulation.variable.StringManipulationVariableNodeFactory": {
            "imports": ["import pandas as pd", "import re"],
            "code": """# String Manipulation on Variable
{variable_name} = {expression_result}""",
            "description": "Manipulate string variables",
            "extract_settings": ["variable_name", "expression"]
        },
        
        # Counting Loop Start
        "org.knime.base.node.flowvariable.loop.CountingLoopStartNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": """# Counting Loop
for _loop_idx in range({start}, {end}, {step}):
    {output_var} = {input_var}.copy()
    _current_iteration = _loop_idx""",
            "description": "Start counting loop",
            "is_loop_start": True,
            "extract_settings": ["start", "end", "step"]
        },
        
        # Column List Loop Start
        "org.knime.base.node.preproc.columnlist.ColumnListLoopStartNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": """# Column List Loop - iterate over columns
for _col_name in {input_var}.columns:
    {output_var} = {input_var}[[_col_name]]
    _current_column = _col_name""",
            "description": "Loop over column names",
            "is_loop_start": True
        },
        
        # Variable Loop End
        "org.knime.base.node.flowvariable.loop.LoopEndVariableNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": """# Loop End - collect accumulated results
# Results are collected in {output_var}
pass  # Loop termination handled by control flow""",
            "description": "End variable loop",
            "is_loop_end": True
        },
        
        # Create Flow Variable
        "org.knime.base.node.flowvariable.createflowvariable.CreateFlowVariableNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": """# Create new flow variable
{output_var} = {input_var}.copy()
{variable_name} = {value}""",
            "description": "Create new flow variable",
            "extract_settings": ["variable_name", "value"]
        },
        
        # ==================== Flow Control Nodes (Task 2.1) ====================
        # IF Switch
        "org.knime.base.node.switches.ifswitch.IFSwitchNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": """# IF Switch: Routes data based on condition
if {condition}:
    {output_var}_true = {input_var}.copy()
    {output_var}_false = pd.DataFrame()  # Empty
else:
    {output_var}_true = pd.DataFrame()  # Empty
    {output_var}_false = {input_var}.copy()""",
            "description": "Route data based on boolean condition",
            "is_switch": True,
            "extract_settings": ["condition"]
        },
        
        # End IF (merge branches)
        "org.knime.base.node.switches.endifswitch.EndIfSwitchNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": """# End IF: Merge branches
# Takes active branch (non-empty DataFrame)
if not {input_var}_true.empty:
    {output_var} = {input_var}_true
else:
    {output_var} = {input_var}_false""",
            "description": "Merge IF switch branches",
            "is_switch_end": True
        },
        
        # CASE Switch
        "org.knime.base.node.switches.caseswitch.CaseSwitchNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": """# CASE Switch: Route data based on variable value
_case_var = {case_variable}
{output_var}_ports = {{}}  # Dictionary of output ports

match _case_var:
    case {case_values[0]}:
        {output_var}_ports[0] = {input_var}.copy()
    case {case_values[1]}:
        {output_var}_ports[1] = {input_var}.copy()
    case _:
        {output_var}_ports['default'] = {input_var}.copy()""",
            "description": "Route data based on case variable",
            "is_switch": True,
            "extract_settings": ["case_variable", "case_values"]
        },
        
        # End CASE
        "org.knime.base.node.switches.endcaseswitch.EndCaseSwitchNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": """# End CASE: Merge all case branches
{output_var} = pd.concat([port for port in {input_ports} if not port.empty], ignore_index=True)""",
            "description": "Merge CASE switch branches",
            "is_switch_end": True
        },
        
        # Empty Table Switch
        "org.knime.base.node.switches.emptytableswitch.EmptyTableSwitchNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": """# Empty Table Switch: Check if table is empty
if {input_var}.empty:
    {output_var}_empty = {input_var}
    {output_var}_nonempty = pd.DataFrame()
else:
    {output_var}_empty = pd.DataFrame()
    {output_var}_nonempty = {input_var}""",
            "description": "Route based on table emptiness",
            "is_switch": True
        },
        
        # Generic Loop Start
        "org.knime.base.node.meta.looper.loopstart.LoopStartNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": """# Loop Start: Initialize loop over rows
_loop_results = []
for _loop_idx, _loop_row in {input_var}.iterrows():
    {output_var} = pd.DataFrame([_loop_row])
    # Loop body here - accumulated results in _loop_results""",
            "description": "Start loop over DataFrame rows",
            "is_loop_start": True
        },
        
        # Generic Loop End
        "org.knime.base.node.meta.looper.loopend.LoopEndNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": """# Loop End: Aggregate loop results
{output_var} = pd.concat(_loop_results, ignore_index=True) if _loop_results else pd.DataFrame()""",
            "description": "End loop and aggregate results",
            "is_loop_end": True
        },
        
        # Chunk Loop Start (process in batches)
        "org.knime.base.node.meta.looper.chunk.LoopStartChunkNodeFactory": {
            "imports": ["import pandas as pd", "import numpy as np"],
            "code": """# Chunk Loop Start: Process in batches
_chunk_size = {chunk_size}
_chunks = np.array_split({input_var}, max(1, len({input_var}) // _chunk_size))
_chunk_results = []

for _chunk_idx, _chunk in enumerate(_chunks):
    {output_var} = pd.DataFrame(_chunk)
    # Process chunk here""",
            "description": "Start loop processing data in chunks",
            "is_loop_start": True,
            "extract_settings": ["chunk_size"]
        },
        
        # Recursive Loop Start
        "org.knime.base.node.meta.looper.recursive.RecursiveLoopStartNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": """# Recursive Loop Start
_recursive_data = {input_var}.copy()
_max_iterations = {max_iterations}
_iteration = 0

while _iteration < _max_iterations:
    {output_var} = _recursive_data
    # Recursive processing here
    _iteration += 1""",
            "description": "Start recursive loop with max iterations",
            "is_loop_start": True,
            "extract_settings": ["max_iterations"]
        },
        
        # Recursive Loop End
        "org.knime.base.node.meta.looper.recursive.RecursiveLoopEndNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": """# Recursive Loop End: Check condition and decide continue/stop
if {continue_condition}:
    _recursive_data = {input_var}.copy()
    # Continue loop
else:
    {output_var} = {input_var}
    # Exit loop""",
            "description": "End recursive loop based on condition",
            "is_loop_end": True,
            "extract_settings": ["continue_condition"]
        },
        
        # Try/Catch Error
        "org.knime.base.node.util.trycatch.TryCatchNodeFactory": {
            "imports": ["import pandas as pd", "import logging"],
            "code": """# Try/Catch: Handle potential errors
logger = logging.getLogger(__name__)
try:
    {output_var} = {input_var}.copy()
    # Protected code here
    {output_var}_error = pd.DataFrame()
except Exception as e:
    logger.warning(f"Error caught: {{e}}")
    {output_var} = pd.DataFrame()
    {output_var}_error = pd.DataFrame({{'error': [str(e)]}})""",
            "description": "Catch and handle errors",
            "has_error_output": True
        },
        
        # Wait Node (synchronization)
        "org.knime.base.node.flowvariable.wait.WaitNodeFactory": {
            "imports": ["import pandas as pd", "import time"],
            "code": """# Wait Node: Synchronization point
{output_var} = {input_var}.copy()
# All inputs synchronized at this point""",
            "description": "Wait for all inputs before proceeding"
        },
        
        # Table Row to Variable Loop Start (already in Variables, reference)
        "org.knime.base.node.flowvariable.tablerowtovariable.iteration.TableRowToVariableIterationLoopStartNodeFactory": {
            "imports": ["import pandas as pd"],
            "code": """# Table Row to Variable Iteration Loop
for _idx, _row in {input_var}.iterrows():
    _row_vars = _row.to_dict()
    {output_var} = pd.DataFrame([_row])
    # Access row values via _row_vars dict""",
            "description": "Iterate rows as variables",
            "is_loop_start": True
        },
        
        # ==================== Database Nodes (Task 2.2) ====================
        # Generic DB Reader
        "org.knime.database.node.reader.DBReaderNodeFactory": {
            "imports": ["import pandas as pd", "import sqlalchemy"],
            "code": """# DB Reader: Read data from database
from sqlalchemy import create_engine

_engine = create_engine({connection_string})
{output_var} = pd.read_sql({query}, _engine)
_engine.dispose()""",
            "description": "Read data from database using SQL query",
            "extract_settings": ["connection_string", "query"]
        },
        
        # Generic DB Writer
        "org.knime.database.node.writer.DBWriterNodeFactory": {
            "imports": ["import pandas as pd", "import sqlalchemy"],
            "code": """# DB Writer: Write data to database
from sqlalchemy import create_engine

_engine = create_engine({connection_string})
{input_var}.to_sql(
    name={table_name},
    con=_engine,
    if_exists={if_exists},  # 'fail', 'replace', 'append'
    index=False
)
_engine.dispose()
{output_var} = {input_var}  # Pass through""",
            "description": "Write DataFrame to database table",
            "extract_settings": ["connection_string", "table_name", "if_exists"]
        },
        
        # DB Query Reader (execute query, return results)
        "org.knime.database.node.query.DBQueryReaderNodeFactory": {
            "imports": ["import pandas as pd", "import sqlalchemy"],
            "code": """# DB Query Reader: Execute SQL query and return results
from sqlalchemy import create_engine, text

_engine = create_engine({connection_string})
with _engine.connect() as _conn:
    {output_var} = pd.read_sql(text({query}), _conn)
_engine.dispose()""",
            "description": "Execute SQL query and return DataFrame",
            "extract_settings": ["connection_string", "query"]
        },
        
        # DB Update (execute update/insert/delete)
        "org.knime.database.node.update.DBUpdateNodeFactory": {
            "imports": ["import pandas as pd", "import sqlalchemy"],
            "code": """# DB Update: Execute SQL update statement
from sqlalchemy import create_engine, text

_engine = create_engine({connection_string})
with _engine.connect() as _conn:
    _result = _conn.execute(text({sql_statement}))
    _conn.commit()
    _rows_affected = _result.rowcount
_engine.dispose()
{output_var} = pd.DataFrame({{'rows_affected': [_rows_affected]}})""",
            "description": "Execute SQL update/insert/delete statement",
            "extract_settings": ["connection_string", "sql_statement"]
        },
        
        # MySQL Connector
        "org.knime.database.connectors.MySQLConnectorNodeFactory": {
            "imports": ["import pandas as pd", "import mysql.connector"],
            "code": """# MySQL Connector: Connect and read from MySQL
_conn = mysql.connector.connect(
    host={host},
    database={database},
    user={user},
    password={password},
    port={port}
)
{output_var} = pd.read_sql({query}, _conn)
_conn.close()""",
            "description": "Connect to MySQL database and execute query",
            "extract_settings": ["host", "database", "user", "password", "port", "query"]
        },
        
        # PostgreSQL Connector
        "org.knime.database.connectors.PostgreSQLConnectorNodeFactory": {
            "imports": ["import pandas as pd", "import psycopg2"],
            "code": """# PostgreSQL Connector: Connect and read from PostgreSQL
_conn = psycopg2.connect(
    host={host},
    database={database},
    user={user},
    password={password},
    port={port}
)
{output_var} = pd.read_sql({query}, _conn)
_conn.close()""",
            "description": "Connect to PostgreSQL database and execute query",
            "extract_settings": ["host", "database", "user", "password", "port", "query"]
        },
        
        # SQLite Connector
        "org.knime.database.connectors.SQLiteConnectorNodeFactory": {
            "imports": ["import pandas as pd", "import sqlite3"],
            "code": """# SQLite Connector: Connect and read from SQLite
_conn = sqlite3.connect({database_path})
{output_var} = pd.read_sql({query}, _conn)
_conn.close()""",
            "description": "Connect to SQLite database and execute query",
            "extract_settings": ["database_path", "query"]
        },
        
        # Oracle Connector
        "org.knime.database.connectors.OracleConnectorNodeFactory": {
            "imports": ["import pandas as pd", "import cx_Oracle"],
            "code": """# Oracle Connector: Connect and read from Oracle
_dsn = cx_Oracle.makedsn({host}, {port}, service_name={service_name})
_conn = cx_Oracle.connect(user={user}, password={password}, dsn=_dsn)
{output_var} = pd.read_sql({query}, _conn)
_conn.close()""",
            "description": "Connect to Oracle database and execute query",
            "extract_settings": ["host", "port", "service_name", "user", "password", "query"]
        },
        
        # H2 Database Connector
        "org.knime.database.connectors.H2ConnectorNodeFactory": {
            "imports": ["import pandas as pd", "import jaydebeapi"],
            "code": """# H2 Connector: Connect and read from H2 database
_conn = jaydebeapi.connect(
    'org.h2.Driver',
    {jdbc_url},
    [{user}, {password}]
)
{output_var} = pd.read_sql({query}, _conn)
_conn.close()""",
            "description": "Connect to H2 database and execute query",
            "extract_settings": ["jdbc_url", "user", "password", "query"]
        },
        
        # SQL Server Connector
        "org.knime.database.connectors.MSSQLConnectorNodeFactory": {
            "imports": ["import pandas as pd", "import pyodbc"],
            "code": """# SQL Server Connector: Connect and read from SQL Server
_conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={{{host}}};DATABASE={{{database}}};UID={{{user}}};PWD={{{password}}}"
_conn = pyodbc.connect(_conn_str)
{output_var} = pd.read_sql({query}, _conn)
_conn.close()""",
            "description": "Connect to SQL Server and execute query",
            "extract_settings": ["host", "database", "user", "password", "query"]
        },
        
        # DB Table Creator
        "org.knime.database.node.tablecreator.DBTableCreatorNodeFactory": {
            "imports": ["import pandas as pd", "import sqlalchemy"],
            "code": """# DB Table Creator: Create new table in database
from sqlalchemy import create_engine

_engine = create_engine({connection_string})
{input_var}.head(0).to_sql(
    name={table_name},
    con=_engine,
    if_exists='fail',
    index=False
)
_engine.dispose()
{output_var} = pd.DataFrame({{'table_created': [{table_name}]}})""",
            "description": "Create new table in database from DataFrame schema",
            "extract_settings": ["connection_string", "table_name"]
        },
        
        # DB Delete (parameterized delete)
        "org.knime.database.node.delete.DBDeleteNodeFactory": {
            "imports": ["import pandas as pd", "import sqlalchemy"],
            "code": """# DB Delete: Delete rows based on DataFrame values
from sqlalchemy import create_engine, text

_engine = create_engine({connection_string})
_deleted_count = 0
with _engine.connect() as _conn:
    for _idx, _row in {input_var}.iterrows():
        _result = _conn.execute(text({delete_query}), _row.to_dict())
        _deleted_count += _result.rowcount
    _conn.commit()
_engine.dispose()
{output_var} = pd.DataFrame({{'rows_deleted': [_deleted_count]}})""",
            "description": "Delete rows from database based on DataFrame",
            "extract_settings": ["connection_string", "delete_query"]
        },
        
        # DB Loader (bulk insert)
        "org.knime.database.node.loader.DBLoaderNodeFactory": {
            "imports": ["import pandas as pd", "import sqlalchemy"],
            "code": """# DB Loader: Bulk load data into database
from sqlalchemy import create_engine

_engine = create_engine({connection_string})
{input_var}.to_sql(
    name={table_name},
    con=_engine,
    if_exists='append',
    index=False,
    method='multi',  # Bulk insert optimization
    chunksize=1000
)
_engine.dispose()
{output_var} = pd.DataFrame({{'rows_loaded': [len({input_var})]}})""",
            "description": "Bulk load DataFrame into database table",
            "extract_settings": ["connection_string", "table_name"]
        },
    }
    
    # Alternative factory patterns (partial matching)
    FACTORY_PATTERNS: Dict[str, str] = {
        "csvreader": "org.knime.base.node.io.csvreader.CSVReaderNodeFactory",
        "csvwriter": "org.knime.base.node.io.csvwriter.CSVWriterNodeFactory",
        "columnfilter": "org.knime.base.node.preproc.filter.column.DataColumnSpecFilterNodeFactory",
        "rowfilter": "org.knime.base.node.preproc.filter.row.RowFilterNodeFactory",
        "groupby": "org.knime.base.node.preproc.groupby.GroupByNodeFactory",
        "joiner": "org.knime.base.node.preproc.joiner.JoinerNodeFactory",
        "concatenate": "org.knime.base.node.preproc.append.row.AppendedRowsNodeFactory",
        "mathformula": "org.knime.base.node.preproc.mathformula.MathFormulaNodeFactory",
        "normalizer": "org.knime.base.node.preproc.normalize.NormalizeNodeFactory",
        "decisiontree": "org.knime.base.node.mine.decisiontree2.learner2.DecTreeLearnerNodeFactory2",
        "kmeans": "org.knime.base.node.mine.cluster.kmeans.KMeansNodeFactory",
        "sorter": "org.knime.base.node.preproc.sorter.SorterNodeFactory",
        "ruleengine": "org.knime.base.node.rules.engine.RuleEngineNodeFactory",
        # Conversion patterns (Task 1.2)
        "numbertostring": "org.knime.base.node.preproc.pmml.numbertostring.NumberToStringNodeFactory",
        "stringtonumber": "org.knime.base.node.preproc.pmml.stringtonumber.StringToNumberNodeFactory",
        "rounddouble": "org.knime.base.node.preproc.rounddouble.RoundDoubleNodeFactory",
        "datetimetostring": "org.knime.time.node.convert.datetimetostring.DateTimeToStringNodeFactory",
        "stringtodatetime": "org.knime.time.node.convert.stringtodatetime.StringToDateTimeNodeFactory",
        "doubletoint": "org.knime.base.node.preproc.doubletoint.DoubleToIntNodeFactory",
        "inttodouble": "org.knime.base.node.preproc.inttodouble.IntToDoubleNodeFactory",
        "columntypechanger": "org.knime.base.node.preproc.columnTypeChange.ColumnTypeChangerNodeFactory",
        # Variable patterns (Task 1.3)
        "tablerowtovar": "org.knime.base.node.preproc.table.rowtovar.TableRowToVariableNodeFactory2",
        "vartotablerow": "org.knime.base.node.flowvariable.variabletotablerow.VariableToTableRowNodeFactory2",
        "stringmanipulation": "org.knime.base.node.flowvariable.stringmanipulation.edit.StringManipulationEditVariableNodeFactory",
        "javasnippet": "org.knime.ext.javasnippet.node.JavaSnippetNodeFactory",
        "countingloop": "org.knime.base.node.flowvariable.loop.CountingLoopStartNodeFactory",
        "columnlistloop": "org.knime.base.node.preproc.columnlist.ColumnListLoopStartNodeFactory",
        "loopend": "org.knime.base.node.flowvariable.loop.LoopEndVariableNodeFactory",
        "createflowvar": "org.knime.base.node.flowvariable.createflowvariable.CreateFlowVariableNodeFactory",
        # Flow Control patterns (Task 2.1)
        "ifswitch": "org.knime.base.node.switches.ifswitch.IFSwitchNodeFactory",
        "endif": "org.knime.base.node.switches.endifswitch.EndIfSwitchNodeFactory",
        "caseswitch": "org.knime.base.node.switches.caseswitch.CaseSwitchNodeFactory",
        "endcase": "org.knime.base.node.switches.endcaseswitch.EndCaseSwitchNodeFactory",
        "emptytableswitch": "org.knime.base.node.switches.emptytableswitch.EmptyTableSwitchNodeFactory",
        "loopstart": "org.knime.base.node.meta.looper.loopstart.LoopStartNodeFactory",
        "loopend": "org.knime.base.node.meta.looper.loopend.LoopEndNodeFactory",
        "chunkloop": "org.knime.base.node.meta.looper.chunk.LoopStartChunkNodeFactory",
        "recursiveloop": "org.knime.base.node.meta.looper.recursive.RecursiveLoopStartNodeFactory",
        "trycatch": "org.knime.base.node.util.trycatch.TryCatchNodeFactory",
        "wait": "org.knime.base.node.flowvariable.wait.WaitNodeFactory",
        # Database patterns (Task 2.2)
        "dbreader": "org.knime.database.node.reader.DBReaderNodeFactory",
        "dbwriter": "org.knime.database.node.writer.DBWriterNodeFactory",
        "dbquery": "org.knime.database.node.query.DBQueryReaderNodeFactory",
        "dbupdate": "org.knime.database.node.update.DBUpdateNodeFactory",
        "mysql": "org.knime.database.connectors.MySQLConnectorNodeFactory",
        "postgresql": "org.knime.database.connectors.PostgreSQLConnectorNodeFactory",
        "sqlite": "org.knime.database.connectors.SQLiteConnectorNodeFactory",
        "oracle": "org.knime.database.connectors.OracleConnectorNodeFactory",
        "h2": "org.knime.database.connectors.H2ConnectorNodeFactory",
        "mssql": "org.knime.database.connectors.MSSQLConnectorNodeFactory",
        "dbtablecreator": "org.knime.database.node.tablecreator.DBTableCreatorNodeFactory",
        "dbdelete": "org.knime.database.node.delete.DBDeleteNodeFactory",
        "dbloader": "org.knime.database.node.loader.DBLoaderNodeFactory",
    }
    
    def get_template(self, factory_class: str) -> Optional[Dict[str, Any]]:
        """
        Get template for a factory class.
        
        Args:
            factory_class: Full KNIME factory class name
            
        Returns:
            Template dictionary or None if not found
        """
        # Direct match in base templates
        if factory_class in self.TEMPLATES:
            return self.TEMPLATES[factory_class]
        
        # Check extended templates
        extended = self._get_extended_templates()
        if extended and factory_class in extended:
            return extended[factory_class]
        
        # Try partial matching
        factory_lower = factory_class.lower()
        for pattern, mapped_factory in self.FACTORY_PATTERNS.items():
            if pattern in factory_lower:
                if mapped_factory in self.TEMPLATES:
                    return self.TEMPLATES[mapped_factory]
        
        return None
    
    def _get_extended_templates(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Lazy load extended templates."""
        global _extended_templates
        if _extended_templates is None:
            try:
                from app.services.generator.extended_templates import get_all_extended_templates
                _extended_templates = get_all_extended_templates()
            except ImportError:
                _extended_templates = {}
        return _extended_templates
    
    def is_supported(self, factory_class: str) -> bool:
        """Check if a factory class has a template."""
        return self.get_template(factory_class) is not None
    
    def get_supported_nodes(self) -> List[str]:
        """Get list of supported factory classes."""
        return list(self.TEMPLATES.keys())
    
    def generate_code(
        self,
        factory_class: str,
        input_var: str,
        output_var: str,
        settings: Dict[str, Any]
    ) -> Optional[Tuple[str, List[str]]]:
        """
        Generate Python code from template.
        
        Args:
            factory_class: KNIME factory class
            input_var: Input variable name
            output_var: Output variable name
            settings: Node settings/configuration
            
        Returns:
            Tuple of (code, imports) or None if not supported
        """
        # Check for scripting nodes first (Python/R with embedded code)
        try:
            from app.services.catalog.node_templates.scripting_templates import (
                is_scripting_node,
                handle_scripting_node
            )
            if is_scripting_node(factory_class):
                code, imports = handle_scripting_node(
                    factory_class, settings, input_var, output_var
                )
                return code, list(imports)
        except ImportError:
            pass  # Scripting templates not available
        except Exception as e:
            logger.warning(f"Error handling scripting node {factory_class}: {e}")
        
        template = self.get_template(factory_class)
        if template is None:
            return None
        
        try:
            code = template['code']
            imports = template['imports']
            
            # Replace standard placeholders
            code = code.replace('{input_var}', input_var)
            code = code.replace('{output_var}', output_var)
            
            # Replace settings-based placeholders
            code = self._replace_settings_placeholders(code, settings)
            
            return code, imports
        except Exception as e:
            logger.warning(f"Error generating code for {factory_class}: {e}")
            return None
    
    def _replace_settings_placeholders(self, code: str, settings: Dict) -> str:
        """Replace placeholders with values from settings."""
        # Parse JEP expressions if present
        expression = settings.get('expression', '')
        if expression and '$' in expression:
            try:
                from app.services.expression import convert as jep_convert
                expression = jep_convert(expression, settings.get('left_var', 'df_input'))
            except Exception:
                expression = "''"  # Fallback
        elif not expression:
            expression = "''"
        
        # Common replacements
        replacements = {
            '{file_path}': settings.get('file_path', 'data.csv'),
            '{separator}': settings.get('separator', ','),
            '{header}': str(settings.get('header', 0)),
            '{columns}': repr(settings.get('columns', [])),
            '{column_order}': repr(settings.get('column_order', [])),
            '{condition}': settings.get('condition', 'True'),
            '{sort_columns}': repr(settings.get('sort_columns', [])),
            '{ascending}': repr(settings.get('ascending', True)),
            '{group_columns}': repr(settings.get('group_columns', [])),
            '{agg_dict}': repr(settings.get('agg_dict', {})),
            '{join_columns}': repr(settings.get('join_columns', [])),
            '{join_type}': settings.get('join_type', 'inner'),
            '{left_columns}': repr(settings.get('left_columns', [])),
            '{right_columns}': repr(settings.get('right_columns', [])),
            '{left_var}': settings.get('left_var', 'df_left'),
            '{right_var}': settings.get('right_var', 'df_right'),
            '{input_vars}': ', '.join(settings.get('input_vars', ['df'])),
            '{new_column}': settings.get('new_column', 'result'),
            '{expression}': expression,
            '{column}': settings.get('column', 'column'),
            '{replace_dict}': repr(settings.get('replace_dict', {})),
            '{fill_value}': repr(settings.get('fill_value', 0)),
            '{method}': settings.get('method', 'ffill'),
            '{unit}': settings.get('unit', 'days'),
            '{col1}': settings.get('col1', 'date1'),
            '{col2}': settings.get('col2', 'date2'),
            '{n_samples}': str(settings.get('n_samples', 100)),
            '{rename_mapping}': repr(settings.get('rename_mapping', {})),
            '{feature_columns}': repr(settings.get('feature_columns', [])),
            '{target_column}': settings.get('target_column', 'target'),
            '{max_depth}': str(settings.get('max_depth', 5)),
            '{n_clusters}': str(settings.get('n_clusters', 3)),
            '{sheet}': str(settings.get('sheet', 0)),
            '{conditions}': repr(settings.get('conditions', [])),
            '{values}': repr(settings.get('values', [])),
            '{default}': repr(settings.get('default', None)),
            '{output_column}': settings.get('output_column', 'result'),
            '{value}': repr(settings.get('value', '')),
            '{variables}': settings.get('variables', 'row_vars'),
            '{precision}': str(settings.get('precision', 2)),
        }
        
        for placeholder, value in replacements.items():
            code = code.replace(placeholder, str(value))
        
        return code
