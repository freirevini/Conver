"""
Node Registry - Catalog of KNIME nodes with Python mappings.

Provides:
- Mapping of 200+ KNIME nodes to Python equivalents
- Template selection for code generation
- Fallback level determination
- Node categorization
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from app.models.ir_models import FallbackLevel, NodeCategory

logger = logging.getLogger(__name__)


@dataclass
class PythonMapping:
    """Mapping from a KNIME node to Python code."""
    # Python libraries required
    imports: List[str] = field(default_factory=list)
    # Template name (Jinja2 template file)
    template: Optional[str] = None
    # Inline code generator function
    code_generator: Optional[Callable[[Dict[str, Any]], str]] = None
    # Fallback level
    fallback_level: FallbackLevel = FallbackLevel.TEMPLATE_EXACT
    # Additional dependencies
    pip_packages: List[str] = field(default_factory=list)
    # Notes for manual implementation
    notes: Optional[str] = None


@dataclass
class NodeDefinition:
    """Complete definition of a KNIME node."""
    # Node identification
    factory_class: str
    display_name: str
    category: NodeCategory
    # Alternative names (for matching)
    aliases: List[str] = field(default_factory=list)
    # Python mapping
    mapping: PythonMapping = field(default_factory=PythonMapping)
    # Input/output port info
    input_ports: int = 1
    output_ports: int = 1
    # Supports flow variables
    supports_flow_vars: bool = True


class NodeRegistry:
    """
    Central registry of all KNIME nodes with Python mappings.
    
    This registry contains definitions for 200+ KNIME nodes,
    categorized and mapped to equivalent Python implementations.
    """
    
    def __init__(self):
        self._nodes: Dict[str, NodeDefinition] = {}
        self._aliases: Dict[str, str] = {}  # alias -> factory_class
        self._initialize_registry()
    
    def _initialize_registry(self) -> None:
        """Initialize the registry with all node definitions."""
        
        # ==================== I/O NODES ====================
        self._register_io_nodes()
        
        # ==================== TRANSFORMATION NODES ====================
        self._register_transform_nodes()
        
        # ==================== FILTER NODES ====================
        self._register_filter_nodes()
        
        # ==================== AGGREGATION NODES ====================
        self._register_aggregation_nodes()
        
        # ==================== JOINING NODES ====================
        self._register_joining_nodes()
        
        # ==================== MACHINE LEARNING NODES ====================
        self._register_ml_nodes()
        
        # ==================== FLOW CONTROL NODES ====================
        self._register_flow_control_nodes()
        
        # ==================== DATABASE NODES ====================
        self._register_database_nodes()
        
        # ==================== DATETIME NODES ====================
        self._register_datetime_nodes()
        
        # ==================== TEXT NODES ====================
        self._register_text_nodes()
        
        # ==================== STATISTICS NODES ====================
        self._register_statistics_nodes()
        
        logger.info(f"Node registry initialized with {len(self._nodes)} nodes")
    
    def _register(self, node: NodeDefinition) -> None:
        """Register a node definition."""
        self._nodes[node.factory_class] = node
        
        # Register aliases
        for alias in node.aliases:
            self._aliases[alias.lower()] = node.factory_class
        
        # Also register display name as alias
        self._aliases[node.display_name.lower()] = node.factory_class
    
    def _register_io_nodes(self) -> None:
        """Register I/O nodes."""
        
        # CSV Reader
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.io.csvreader.CSVReaderNodeFactory",
            display_name="CSV Reader",
            category=NodeCategory.IO,
            aliases=["csv reader", "read csv", "csvreader"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="io/csv_reader.py.j2",
                pip_packages=["pandas"],
            ),
        ))
        
        # CSV Writer
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.io.csvwriter.CSVWriterNodeFactory",
            display_name="CSV Writer",
            category=NodeCategory.IO,
            aliases=["csv writer", "write csv", "csvwriter"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="io/csv_writer.py.j2",
                pip_packages=["pandas"],
            ),
            input_ports=1,
            output_ports=0,
        ))
        
        # Excel Reader
        self._register(NodeDefinition(
            factory_class="org.knime.ext.poi3.node.io.filehandling.excel.reader.ExcelTableReaderNodeFactory",
            display_name="Excel Reader",
            category=NodeCategory.IO,
            aliases=["excel reader", "read excel", "xlsx reader"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="io/excel_reader.py.j2",
                pip_packages=["pandas", "openpyxl"],
            ),
        ))
        
        # Excel Writer
        self._register(NodeDefinition(
            factory_class="org.knime.ext.poi3.node.io.filehandling.excel.writer.ExcelTableWriterNodeFactory",
            display_name="Excel Writer",
            category=NodeCategory.IO,
            aliases=["excel writer", "write excel", "xlsx writer"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="io/excel_writer.py.j2",
                pip_packages=["pandas", "openpyxl"],
            ),
            input_ports=1,
            output_ports=0,
        ))
        
        # Parquet Reader
        self._register(NodeDefinition(
            factory_class="org.knime.parquet.node.read.ParquetReaderNodeFactory",
            display_name="Parquet Reader",
            category=NodeCategory.IO,
            aliases=["parquet reader", "read parquet"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="io/parquet_reader.py.j2",
                pip_packages=["pandas", "pyarrow"],
            ),
        ))
        
        # JSON Reader
        self._register(NodeDefinition(
            factory_class="org.knime.json.node.reader.JSONReaderNodeFactory",
            display_name="JSON Reader",
            category=NodeCategory.IO,
            aliases=["json reader", "read json"],
            mapping=PythonMapping(
                imports=["import pandas as pd", "import json"],
                template="io/json_reader.py.j2",
                pip_packages=["pandas"],
            ),
        ))
        
        # Table Creator
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.io.tablecreator.TableCreator2NodeFactory",
            display_name="Table Creator",
            category=NodeCategory.IO,
            aliases=["table creator", "create table"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="io/table_creator.py.j2",
            ),
            input_ports=0,
        ))
    
    def _register_transform_nodes(self) -> None:
        """Register transformation nodes."""
        
        # Column Filter
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.preproc.filter.column.DataColumnSpecFilterNodeFactory",
            display_name="Column Filter",
            category=NodeCategory.TRANSFORM,
            aliases=["column filter", "filter columns", "select columns"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="transform/column_filter.py.j2",
            ),
        ))
        
        # Column Rename
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.preproc.rename.RenameNodeFactory",
            display_name="Column Rename",
            category=NodeCategory.TRANSFORM,
            aliases=["column rename", "rename columns", "rename"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="transform/column_rename.py.j2",
            ),
        ))
        
        # Column Resorter
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.preproc.columnreorder.DataColumnReorderNodeFactory",
            display_name="Column Resorter",
            category=NodeCategory.TRANSFORM,
            aliases=["column resorter", "reorder columns", "column reorder"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="transform/column_resorter.py.j2",
            ),
        ))
        
        # Math Formula
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.preproc.pmml.stringtonumber.MathFormulaNodeFactory",
            display_name="Math Formula",
            category=NodeCategory.TRANSFORM,
            aliases=["math formula", "formula", "calculate"],
            mapping=PythonMapping(
                imports=["import pandas as pd", "import numpy as np"],
                template="transform/math_formula.py.j2",
            ),
        ))
        
        # Rule Engine
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.rules.engine.RuleEngineNodeFactory",
            display_name="Rule Engine",
            category=NodeCategory.TRANSFORM,
            aliases=["rule engine", "rules", "business rules"],
            mapping=PythonMapping(
                imports=["import pandas as pd", "import numpy as np"],
                template="transform/rule_engine.py.j2",
            ),
        ))
        
        # String Manipulation
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.preproc.stringmanipulation.StringManipulationNodeFactory",
            display_name="String Manipulation",
            category=NodeCategory.TRANSFORM,
            aliases=["string manipulation", "string functions"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="transform/string_manipulation.py.j2",
            ),
        ))
        
        # Double to Int
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.preproc.double2int.DoubleToIntNodeFactory",
            display_name="Double To Int",
            category=NodeCategory.TRANSFORM,
            aliases=["double to int", "float to int", "convert to int"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="transform/double_to_int.py.j2",
            ),
        ))
        
        # Missing Value
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.preproc.pmml.missingval.MissingValueHandlerNodeFactory",
            display_name="Missing Value",
            category=NodeCategory.TRANSFORM,
            aliases=["missing value", "handle missing", "fill na"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="transform/missing_value.py.j2",
            ),
        ))
        
        # Normalizer
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.preproc.normalize3.Normalizer3NodeFactory",
            display_name="Normalizer",
            category=NodeCategory.TRANSFORM,
            aliases=["normalizer", "normalize", "scale"],
            mapping=PythonMapping(
                imports=["import pandas as pd", "from sklearn.preprocessing import MinMaxScaler, StandardScaler"],
                template="transform/normalizer.py.j2",
                pip_packages=["scikit-learn"],
            ),
            output_ports=2,  # Data + Model
        ))
    
    def _register_filter_nodes(self) -> None:
        """Register filter nodes."""
        
        # Row Filter
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.preproc.filter.row3.RowFilterNodeFactory",
            display_name="Row Filter",
            category=NodeCategory.FILTER,
            aliases=["row filter", "filter rows", "where"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="filter/row_filter.py.j2",
            ),
        ))
        
        # Row Splitter
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.preproc.filter.row.RowSplitterNodeFactory",
            display_name="Row Splitter",
            category=NodeCategory.FILTER,
            aliases=["row splitter", "split rows"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="filter/row_splitter.py.j2",
            ),
            output_ports=2,
        ))
        
        # Duplicate Row Filter
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.preproc.duplicaterowfilter.DuplicateRowFilterNodeFactory",
            display_name="Duplicate Row Filter",
            category=NodeCategory.FILTER,
            aliases=["duplicate row filter", "remove duplicates", "deduplicate"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="filter/duplicate_row_filter.py.j2",
            ),
        ))
        
        # RowID
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.preproc.rowkey.RowKeyNodeFactory",
            display_name="RowID",
            category=NodeCategory.FILTER,
            aliases=["rowid", "row id", "set row id"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="filter/row_id.py.j2",
            ),
        ))
        
        # Top k Row Filter
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.preproc.topk.TopKSelectorNodeFactory",
            display_name="Top k Row Filter",
            category=NodeCategory.FILTER,
            aliases=["top k", "top n", "head", "limit"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="filter/top_k.py.j2",
            ),
        ))
    
    def _register_aggregation_nodes(self) -> None:
        """Register aggregation nodes."""
        
        # GroupBy
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.preproc.groupby.GroupByNodeFactory",
            display_name="GroupBy",
            category=NodeCategory.AGGREGATION,
            aliases=["groupby", "group by", "aggregate"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="aggregation/groupby.py.j2",
            ),
        ))
        
        # Pivoting
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.preproc.pivot.Pivot2NodeFactory",
            display_name="Pivoting",
            category=NodeCategory.AGGREGATION,
            aliases=["pivot", "pivoting", "pivot table"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="aggregation/pivot.py.j2",
            ),
        ))
        
        # Unpivoting
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.preproc.unpivot.UnpivotNodeFactory",
            display_name="Unpivoting",
            category=NodeCategory.AGGREGATION,
            aliases=["unpivot", "unpivoting", "melt"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="aggregation/unpivot.py.j2",
            ),
        ))
        
        # Column Aggregator
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.preproc.colcombine2.ColumnAggregatorNodeFactory",
            display_name="Column Aggregator",
            category=NodeCategory.AGGREGATION,
            aliases=["column aggregator", "aggregate columns"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="aggregation/column_aggregator.py.j2",
            ),
        ))
    
    def _register_joining_nodes(self) -> None:
        """Register joining nodes."""
        
        # Joiner
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.preproc.joiner.Joiner3NodeFactory",
            display_name="Joiner",
            category=NodeCategory.JOINING,
            aliases=["joiner", "join", "merge"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="joining/joiner.py.j2",
            ),
            input_ports=2,
        ))
        
        # Cross Joiner
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.preproc.crossjoiner.CrossJoinerNodeFactory",
            display_name="Cross Joiner",
            category=NodeCategory.JOINING,
            aliases=["cross joiner", "cross join", "cartesian"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="joining/cross_joiner.py.j2",
            ),
            input_ports=2,
        ))
        
        # Concatenate
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.preproc.append.row.AppendedRowsNodeFactory",
            display_name="Concatenate",
            category=NodeCategory.JOINING,
            aliases=["concatenate", "concat", "append rows", "union"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="joining/concatenate.py.j2",
            ),
            input_ports=2,
        ))
        
        # Column Appender
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.preproc.columnappend.ColumnAppenderNodeFactory",
            display_name="Column Appender",
            category=NodeCategory.JOINING,
            aliases=["column appender", "append columns"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="joining/column_appender.py.j2",
            ),
            input_ports=2,
        ))
    
    def _register_ml_nodes(self) -> None:
        """Register machine learning nodes."""
        
        # Decision Tree Learner
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.mine.decisiontree2.learner.DecisionTreeLearnerNodeFactory",
            display_name="Decision Tree Learner",
            category=NodeCategory.MACHINE_LEARNING,
            aliases=["decision tree learner", "decision tree"],
            mapping=PythonMapping(
                imports=["from sklearn.tree import DecisionTreeClassifier"],
                template="ml/decision_tree_learner.py.j2",
                pip_packages=["scikit-learn"],
            ),
            output_ports=2,  # Model + Statistics
        ))
        
        # Decision Tree Predictor
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.mine.decisiontree2.predictor.DecisionTreePredictorNodeFactory",
            display_name="Decision Tree Predictor",
            category=NodeCategory.MACHINE_LEARNING,
            aliases=["decision tree predictor", "predict decision tree"],
            mapping=PythonMapping(
                imports=["from sklearn.tree import DecisionTreeClassifier"],
                template="ml/decision_tree_predictor.py.j2",
                pip_packages=["scikit-learn"],
            ),
            input_ports=2,  # Model + Data
        ))
        
        # Random Forest Learner
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.mine.treeensemble2.node.learner.classification.TreeEnsembleClassificationLearnerNodeFactory",
            display_name="Random Forest Learner",
            category=NodeCategory.MACHINE_LEARNING,
            aliases=["random forest learner", "random forest"],
            mapping=PythonMapping(
                imports=["from sklearn.ensemble import RandomForestClassifier"],
                template="ml/random_forest_learner.py.j2",
                pip_packages=["scikit-learn"],
            ),
            output_ports=2,
        ))
        
        # Linear Regression Learner
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.mine.regression.linear2.learner.LinearRegressionLearnerNodeFactory",
            display_name="Linear Regression Learner",
            category=NodeCategory.MACHINE_LEARNING,
            aliases=["linear regression learner", "linear regression"],
            mapping=PythonMapping(
                imports=["from sklearn.linear_model import LinearRegression"],
                template="ml/linear_regression_learner.py.j2",
                pip_packages=["scikit-learn"],
            ),
            output_ports=2,
        ))
        
        # k-Means
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.mine.cluster.kmeans.KMeansNodeFactory",
            display_name="k-Means",
            category=NodeCategory.MACHINE_LEARNING,
            aliases=["kmeans", "k-means", "clustering"],
            mapping=PythonMapping(
                imports=["from sklearn.cluster import KMeans"],
                template="ml/kmeans.py.j2",
                pip_packages=["scikit-learn"],
            ),
            output_ports=2,
        ))
        
        # Partitioning
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.preproc.partition.PartitionNodeFactory",
            display_name="Partitioning",
            category=NodeCategory.MACHINE_LEARNING,
            aliases=["partitioning", "train test split", "split"],
            mapping=PythonMapping(
                imports=["from sklearn.model_selection import train_test_split"],
                template="ml/partitioning.py.j2",
                pip_packages=["scikit-learn"],
            ),
            output_ports=2,
        ))
        
        # Scorer
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.mine.scorer.accuracy.AccuracyScorerNodeFactory",
            display_name="Scorer",
            category=NodeCategory.MACHINE_LEARNING,
            aliases=["scorer", "accuracy", "evaluate"],
            mapping=PythonMapping(
                imports=["from sklearn.metrics import accuracy_score, confusion_matrix"],
                template="ml/scorer.py.j2",
                pip_packages=["scikit-learn"],
            ),
            output_ports=2,
        ))
    
    def _register_flow_control_nodes(self) -> None:
        """Register flow control nodes."""
        
        # IF Switch
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.switches.ifswitch.IfSwitchNodeFactory",
            display_name="IF Switch",
            category=NodeCategory.FLOW_CONTROL,
            aliases=["if switch", "if", "switch"],
            mapping=PythonMapping(
                imports=[],
                template="flow_control/if_switch.py.j2",
            ),
            output_ports=2,
        ))
        
        # End IF
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.switches.endswitch.EndSwitchNodeFactory",
            display_name="End IF",
            category=NodeCategory.FLOW_CONTROL,
            aliases=["end if", "endif", "merge branches"],
            mapping=PythonMapping(
                imports=[],
                template="flow_control/end_if.py.j2",
            ),
            input_ports=2,
        ))
        
        # Empty Table Switch
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.switches.emptytableswitch.EmptyTableSwitchNodeFactory",
            display_name="Empty Table Switch",
            category=NodeCategory.FLOW_CONTROL,
            aliases=["empty table switch", "if empty"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="flow_control/empty_table_switch.py.j2",
            ),
            output_ports=2,
        ))
        
        # Chunk Loop Start
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.meta.looper.chunk.LoopStartChunkNodeFactory",
            display_name="Chunk Loop Start",
            category=NodeCategory.FLOW_CONTROL,
            aliases=["chunk loop start", "loop start chunk"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="flow_control/chunk_loop_start.py.j2",
            ),
        ))
        
        # Loop End
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.meta.looper.LoopEndNodeFactory",
            display_name="Loop End",
            category=NodeCategory.FLOW_CONTROL,
            aliases=["loop end", "end loop"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="flow_control/loop_end.py.j2",
            ),
        ))
        
        # Counting Loop Start
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.meta.looper.count.LoopStartCountNodeFactory",
            display_name="Counting Loop Start",
            category=NodeCategory.FLOW_CONTROL,
            aliases=["counting loop start", "for loop"],
            mapping=PythonMapping(
                imports=[],
                template="flow_control/counting_loop_start.py.j2",
            ),
            input_ports=0,
        ))
    
    def _register_database_nodes(self) -> None:
        """Register database nodes."""
        
        # Database Connector
        self._register(NodeDefinition(
            factory_class="org.knime.database.node.connector.generic.GenericDBConnectorNodeFactory",
            display_name="Database Connector",
            category=NodeCategory.DATABASE,
            aliases=["database connector", "db connector", "connect database"],
            mapping=PythonMapping(
                imports=["from sqlalchemy import create_engine"],
                template="database/db_connector.py.j2",
                pip_packages=["sqlalchemy"],
            ),
            input_ports=0,
        ))
        
        # Database Reader (Legacy)
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.io.database.DBReaderNodeFactory",
            display_name="Database Reader _legacy_",
            category=NodeCategory.DATABASE,
            aliases=["database reader legacy", "db reader legacy"],
            mapping=PythonMapping(
                imports=["import pandas as pd", "from sqlalchemy import create_engine"],
                template="database/db_reader_legacy.py.j2",
                pip_packages=["sqlalchemy", "pandas"],
            ),
        ))
        
        # DB Loader
        self._register(NodeDefinition(
            factory_class="org.knime.database.node.io.load.DBLoaderNodeFactory",
            display_name="DB Loader",
            category=NodeCategory.DATABASE,
            aliases=["db loader", "database loader", "load to db"],
            mapping=PythonMapping(
                imports=["import pandas as pd", "from sqlalchemy import create_engine"],
                template="database/db_loader.py.j2",
                pip_packages=["sqlalchemy", "pandas"],
            ),
            input_ports=2,  # Data + Connection
            output_ports=0,
        ))
        
        # Database Connector (Legacy)
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.io.database.connection.DatabaseConnectionNodeFactory",
            display_name="Database Connector _legacy_",
            category=NodeCategory.DATABASE,
            aliases=["database connector legacy", "db connector legacy"],
            mapping=PythonMapping(
                imports=["from sqlalchemy import create_engine"],
                template="database/db_connector_legacy.py.j2",
                pip_packages=["sqlalchemy"],
            ),
            input_ports=0,
        ))
        
        # Google BigQuery Connector
        self._register(NodeDefinition(
            factory_class="org.knime.google.cloud.bigquery.node.connector.BigQueryConnectorNodeFactory",
            display_name="Google BigQuery Connector",
            category=NodeCategory.DATABASE,
            aliases=["bigquery connector", "google bigquery", "bq connector"],
            mapping=PythonMapping(
                imports=["from google.cloud import bigquery"],
                template="database/bigquery_connector.py.j2",
                pip_packages=["google-cloud-bigquery"],
                fallback_level=FallbackLevel.TEMPLATE_APPROXIMATE,
                notes="Requires Google Cloud credentials setup",
            ),
            input_ports=1,  # Auth
        ))
        
        # Google Authentication (API Key)
        self._register(NodeDefinition(
            factory_class="org.knime.google.auth.node.apikey.GoogleApiKeyAuthenticatorNodeFactory",
            display_name="Google Authentication _API Key_",
            category=NodeCategory.DATABASE,
            aliases=["google auth", "google api key", "gcp auth"],
            mapping=PythonMapping(
                imports=["import os"],
                template="database/google_auth_api_key.py.j2",
                fallback_level=FallbackLevel.TEMPLATE_APPROXIMATE,
                notes="Requires GOOGLE_API_KEY environment variable",
            ),
            input_ports=0,
        ))
    
    def _register_datetime_nodes(self) -> None:
        """Register datetime nodes."""
        
        # Create Date&Time Range
        self._register(NodeDefinition(
            factory_class="org.knime.time.node.create.createdatetimerange.CreateDateTimeRangeNodeFactory",
            display_name="Create Date_Time Range",
            category=NodeCategory.DATETIME,
            aliases=["create datetime range", "date range", "time range"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="datetime/create_datetime_range.py.j2",
            ),
            input_ports=0,
        ))
        
        # Modify Time
        self._register(NodeDefinition(
            factory_class="org.knime.time.node.manipulate.modifytime.ModifyTimeNodeFactory",
            display_name="Modify Time",
            category=NodeCategory.DATETIME,
            aliases=["modify time", "adjust time", "time manipulation"],
            mapping=PythonMapping(
                imports=["import pandas as pd", "from datetime import timedelta"],
                template="datetime/modify_time.py.j2",
            ),
        ))
        
        # Legacy Date&Time to Date&Time
        self._register(NodeDefinition(
            factory_class="org.knime.time.node.convert.stringtodatetime.StringToDateTimeNodeFactory",
            display_name="Legacy Date_Time to Date_Time",
            category=NodeCategory.DATETIME,
            aliases=["legacy datetime", "convert datetime", "string to datetime"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="datetime/legacy_datetime_convert.py.j2",
            ),
        ))
        
        # Date&Time Difference
        self._register(NodeDefinition(
            factory_class="org.knime.time.node.calculate.datetimedifference.DateTimeDifferenceNodeFactory",
            display_name="Date&Time Difference",
            category=NodeCategory.DATETIME,
            aliases=["datetime difference", "time difference", "date diff"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="datetime/datetime_difference.py.j2",
            ),
        ))
    
    def _register_text_nodes(self) -> None:
        """Register text processing nodes."""
        
        # String Replacer
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.preproc.stringreplacer.StringReplacerNodeFactory",
            display_name="String Replacer",
            category=NodeCategory.TEXT,
            aliases=["string replacer", "replace string", "regex replace"],
            mapping=PythonMapping(
                imports=["import pandas as pd", "import re"],
                template="text/string_replacer.py.j2",
            ),
        ))
        
        # Cell Splitter
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.preproc.cellsplit.CellSplitterNodeFactory",
            display_name="Cell Splitter",
            category=NodeCategory.TEXT,
            aliases=["cell splitter", "split cells", "tokenize"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="text/cell_splitter.py.j2",
            ),
        ))
        
        # String to Number
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.preproc.pmml.stringtonumber.StringToNumberNodeFactory",
            display_name="String to Number",
            category=NodeCategory.TEXT,
            aliases=["string to number", "parse number"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="text/string_to_number.py.j2",
            ),
        ))
    
    def _register_statistics_nodes(self) -> None:
        """Register statistics nodes."""
        
        # Statistics
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.stats.statistics3.Statistics3NodeFactory",
            display_name="Statistics",
            category=NodeCategory.STATISTICS,
            aliases=["statistics", "stats", "describe"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="statistics/statistics.py.j2",
            ),
            output_ports=2,
        ))
        
        # Correlation
        self._register(NodeDefinition(
            factory_class="org.knime.base.node.stats.correlation.compute2.CorrelationCompute2NodeFactory",
            display_name="Linear Correlation",
            category=NodeCategory.STATISTICS,
            aliases=["correlation", "linear correlation", "pearson"],
            mapping=PythonMapping(
                imports=["import pandas as pd"],
                template="statistics/correlation.py.j2",
            ),
        ))
    
    # ==================== QUERY METHODS ====================
    
    def get_node(self, identifier: str) -> Optional[NodeDefinition]:
        """
        Get a node definition by factory class or alias.
        
        Args:
            identifier: Factory class or node alias
            
        Returns:
            NodeDefinition or None if not found
        """
        # Try direct factory class lookup
        if identifier in self._nodes:
            return self._nodes[identifier]
        
        # Try alias lookup
        identifier_lower = identifier.lower()
        if identifier_lower in self._aliases:
            factory = self._aliases[identifier_lower]
            return self._nodes.get(factory)
        
        # Try partial matching
        for alias, factory in self._aliases.items():
            if identifier_lower in alias or alias in identifier_lower:
                return self._nodes.get(factory)
        
        return None
    
    def get_mapping(self, identifier: str) -> Optional[PythonMapping]:
        """Get Python mapping for a node."""
        node = self.get_node(identifier)
        return node.mapping if node else None
    
    def get_template(self, identifier: str) -> Optional[str]:
        """Get template name for a node."""
        mapping = self.get_mapping(identifier)
        return mapping.template if mapping else None
    
    def get_fallback_level(self, identifier: str) -> FallbackLevel:
        """Get fallback level for a node."""
        mapping = self.get_mapping(identifier)
        if mapping:
            return mapping.fallback_level
        return FallbackLevel.STUB
    
    def get_all_nodes(self) -> List[NodeDefinition]:
        """Get all registered node definitions."""
        return list(self._nodes.values())
    
    def get_nodes_by_category(self, category: NodeCategory) -> List[NodeDefinition]:
        """Get all nodes in a specific category."""
        return [n for n in self._nodes.values() if n.category == category]
    
    def get_supported_count(self) -> int:
        """Get count of fully supported nodes."""
        return sum(
            1 for n in self._nodes.values()
            if n.mapping.fallback_level == FallbackLevel.TEMPLATE_EXACT
        )
    
    def search(self, query: str) -> List[NodeDefinition]:
        """Search nodes by name or alias."""
        query_lower = query.lower()
        results = []
        
        for node in self._nodes.values():
            if query_lower in node.display_name.lower():
                results.append(node)
            elif any(query_lower in alias for alias in node.aliases):
                results.append(node)
        
        return results


# Global registry instance
_registry: Optional[NodeRegistry] = None


def get_node_registry() -> NodeRegistry:
    """Get the global node registry instance."""
    global _registry
    if _registry is None:
        _registry = NodeRegistry()
    return _registry
