"""
Test Factories and Fixtures for KNIME Transpiler Tests.

Provides:
- Node instance factories
- Workflow mock factories
- Configuration generators
- Template test helpers
"""
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class MockNode:
    """Mock KNIME node for testing."""
    id: str
    name: str
    factory: str
    settings: Dict[str, Any] = field(default_factory=dict)
    category: str = "Other"
    is_metanode: bool = False
    depth: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "factory": self.factory,
            "factory_class": self.factory,
            "settings": self.settings,
            "category": self.category,
            "is_metanode": self.is_metanode,
            "depth": self.depth
        }


@dataclass
class MockConnection:
    """Mock KNIME connection for testing."""
    source_id: str
    dest_id: str
    source_port: int = 1
    dest_port: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "dest_id": self.dest_id,
            "source_port": self.source_port,
            "dest_port": self.dest_port
        }


@dataclass
class MockWorkflow:
    """Mock KNIME workflow for testing."""
    name: str
    nodes: List[MockNode] = field(default_factory=list)
    connections: List[MockConnection] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "nodes": [n.to_dict() for n in self.nodes],
            "connections": [c.to_dict() for c in self.connections]
        }


class NodeFactory:
    """Factory for creating test nodes."""
    
    _next_id = 1
    
    @classmethod
    def reset(cls):
        """Reset node ID counter."""
        cls._next_id = 1
    
    @classmethod
    def _get_next_id(cls) -> str:
        node_id = str(cls._next_id)
        cls._next_id += 1
        return node_id
    
    @classmethod
    def csv_reader(cls, file_path: str = "data.csv", **settings) -> MockNode:
        """Create CSV Reader node."""
        return MockNode(
            id=cls._get_next_id(),
            name="CSV Reader",
            factory="org.knime.base.node.io.csvreader.CSVReaderNodeFactory",
            category="IO",
            settings={"file_path": file_path, "separator": ",", "header": 0, **settings}
        )
    
    @classmethod
    def csv_writer(cls, file_path: str = "output.csv", **settings) -> MockNode:
        """Create CSV Writer node."""
        return MockNode(
            id=cls._get_next_id(),
            name="CSV Writer",
            factory="org.knime.base.node.io.csvwriter.CSVWriterNodeFactory",
            category="IO",
            settings={"file_path": file_path, "separator": ",", **settings}
        )
    
    @classmethod
    def column_filter(cls, columns: List[str], **settings) -> MockNode:
        """Create Column Filter node."""
        return MockNode(
            id=cls._get_next_id(),
            name="Column Filter",
            factory="org.knime.base.node.preproc.filter.column.DataColumnSpecFilterNodeFactory",
            category="Manipulation",
            settings={"columns": columns, **settings}
        )
    
    @classmethod
    def row_filter(cls, condition: str = "col > 0", **settings) -> MockNode:
        """Create Row Filter node."""
        return MockNode(
            id=cls._get_next_id(),
            name="Row Filter",
            factory="org.knime.base.node.preproc.filter.row.RowFilterNodeFactory",
            category="Manipulation",
            settings={"condition": condition, **settings}
        )
    
    @classmethod
    def joiner(cls, join_columns: List[str], **settings) -> MockNode:
        """Create Joiner node."""
        return MockNode(
            id=cls._get_next_id(),
            name="Joiner",
            factory="org.knime.base.node.preproc.joiner.Joiner2NodeFactory",
            category="Manipulation",
            settings={"join_columns": join_columns, **settings}
        )
    
    @classmethod
    def groupby(cls, group_columns: List[str], agg_columns: List[str], **settings) -> MockNode:
        """Create GroupBy node."""
        return MockNode(
            id=cls._get_next_id(),
            name="GroupBy",
            factory="org.knime.base.node.preproc.groupby.GroupByNodeFactory",
            category="Manipulation",
            settings={"group_columns": group_columns, "agg_columns": agg_columns, **settings}
        )
    
    @classmethod
    def random_forest_learner(cls, target: str, features: List[str], **settings) -> MockNode:
        """Create Random Forest Learner node."""
        return MockNode(
            id=cls._get_next_id(),
            name="Random Forest Learner",
            factory="org.knime.base.node.mine.treensemble2.node.learner.classification.TreeEnsembleClassificationLearnerNodeFactory3",
            category="Mining",
            settings={
                "target_column": target,
                "feature_columns": features,
                "n_estimators": 100,
                "max_depth": 10,
                **settings
            }
        )
    
    @classmethod
    def random_forest_predictor(cls, features: List[str], **settings) -> MockNode:
        """Create Random Forest Predictor node."""
        return MockNode(
            id=cls._get_next_id(),
            name="Random Forest Predictor",
            factory="org.knime.base.node.mine.treensemble2.node.predictor.classification.TreeEnsembleClassificationPredictorNodeFactory",
            category="Mining",
            settings={"feature_columns": features, **settings}
        )
    
    @classmethod
    def kmeans(cls, n_clusters: int = 3, features: List[str] = None, **settings) -> MockNode:
        """Create K-Means node."""
        return MockNode(
            id=cls._get_next_id(),
            name="k-Means",
            factory="org.knime.base.node.mine.cluster.kmeans.KMeansNodeFactory",
            category="Mining",
            settings={
                "n_clusters": n_clusters,
                "feature_columns": features or [],
                **settings
            }
        )
    
    @classmethod
    def scorer(cls, actual_col: str = "actual", predicted_col: str = "predicted", **settings) -> MockNode:
        """Create Scorer node."""
        return MockNode(
            id=cls._get_next_id(),
            name="Scorer",
            factory="org.knime.base.node.mine.scorer.accuracy.AccuracyScorerNodeFactory",
            category="Mining",
            settings={
                "actual_column": actual_col,
                "predicted_column": predicted_col,
                **settings
            }
        )
    
    @classmethod
    def db_reader(cls, query: str = "SELECT * FROM table", **settings) -> MockNode:
        """Create Database Reader node."""
        return MockNode(
            id=cls._get_next_id(),
            name="DB Reader",
            factory="org.knime.database.node.io.read.DBReaderNodeFactory",
            category="Database",
            settings={"query": query, **settings}
        )
    
    @classmethod
    def generic(cls, name: str, factory: str, category: str = "Other", **settings) -> MockNode:
        """Create a generic node with custom parameters."""
        return MockNode(
            id=cls._get_next_id(),
            name=name,
            factory=factory,
            category=category,
            settings=settings
        )


class WorkflowFactory:
    """Factory for creating test workflows."""
    
    @classmethod
    def empty(cls, name: str = "Test Workflow") -> MockWorkflow:
        """Create empty workflow."""
        NodeFactory.reset()
        return MockWorkflow(name=name)
    
    @classmethod
    def simple_etl(cls) -> MockWorkflow:
        """Create simple ETL workflow: Read -> Filter -> Write."""
        NodeFactory.reset()
        
        reader = NodeFactory.csv_reader("input.csv")
        filter_node = NodeFactory.column_filter(["col1", "col2"])
        writer = NodeFactory.csv_writer("output.csv")
        
        return MockWorkflow(
            name="Simple ETL",
            nodes=[reader, filter_node, writer],
            connections=[
                MockConnection(reader.id, filter_node.id),
                MockConnection(filter_node.id, writer.id)
            ]
        )
    
    @classmethod
    def ml_pipeline(cls) -> MockWorkflow:
        """Create ML training pipeline."""
        NodeFactory.reset()
        
        reader = NodeFactory.csv_reader("train_data.csv")
        rf_learner = NodeFactory.random_forest_learner(
            target="target",
            features=["feat1", "feat2", "feat3"]
        )
        rf_predictor = NodeFactory.random_forest_predictor(
            features=["feat1", "feat2", "feat3"]
        )
        scorer = NodeFactory.scorer()
        
        return MockWorkflow(
            name="ML Pipeline",
            nodes=[reader, rf_learner, rf_predictor, scorer],
            connections=[
                MockConnection(reader.id, rf_learner.id),
                MockConnection(rf_learner.id, rf_predictor.id, source_port=2),
                MockConnection(reader.id, rf_predictor.id),
                MockConnection(rf_predictor.id, scorer.id)
            ]
        )
    
    @classmethod
    def data_aggregation(cls) -> MockWorkflow:
        """Create data aggregation workflow."""
        NodeFactory.reset()
        
        reader = NodeFactory.csv_reader("sales.csv")
        groupby = NodeFactory.groupby(
            group_columns=["region", "product"],
            agg_columns=["amount"]
        )
        writer = NodeFactory.csv_writer("summary.csv")
        
        return MockWorkflow(
            name="Data Aggregation",
            nodes=[reader, groupby, writer],
            connections=[
                MockConnection(reader.id, groupby.id),
                MockConnection(groupby.id, writer.id)
            ]
        )
    
    @classmethod
    def clustering(cls) -> MockWorkflow:
        """Create clustering workflow."""
        NodeFactory.reset()
        
        reader = NodeFactory.csv_reader("customers.csv")
        kmeans = NodeFactory.kmeans(n_clusters=5, features=["age", "income", "score"])
        writer = NodeFactory.csv_writer("clustered.csv")
        
        return MockWorkflow(
            name="Customer Clustering",
            nodes=[reader, kmeans, writer],
            connections=[
                MockConnection(reader.id, kmeans.id),
                MockConnection(kmeans.id, writer.id)
            ]
        )


class ConfigFactory:
    """Factory for creating test configurations."""
    
    @classmethod
    def default_generator_config(cls) -> Dict[str, Any]:
        """Default code generator configuration."""
        return {
            "indent_size": 4,
            "include_comments": True,
            "include_types": True,
            "max_line_length": 88
        }
    
    @classmethod
    def ml_training_config(cls, target: str, features: List[str]) -> Dict[str, Any]:
        """ML training configuration."""
        return {
            "target_column": target,
            "feature_columns": features,
            "test_size": 0.2,
            "random_state": 42
        }
    
    @classmethod
    def db_connection_config(cls, driver: str = "postgresql") -> Dict[str, Any]:
        """Database connection configuration."""
        return {
            "driver": driver,
            "host": "localhost",
            "port": 5432,
            "database": "testdb",
            "username": "test_user"
        }
