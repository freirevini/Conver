#!/usr/bin/env python
"""
ChatKnime Transpiler CLI

Simple command-line tool to convert KNIME workflows (.knwf) to Python.

Usage:
    python transpile.py arquivo.knwf
    python transpile.py arquivo.knwf --output meu_pipeline.py
    python transpile.py arquivo.knwf --no-llm
"""
import sys
import argparse
import zipfile
import tempfile
import shutil
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))


def extract_knwf(knwf_path: Path, extract_dir: Path) -> Path:
    """Extract .knwf file and return path to workflow folder."""
    with zipfile.ZipFile(knwf_path, 'r') as zf:
        zf.extractall(extract_dir)
    
    # Find workflow.knime in extracted content
    for item in extract_dir.rglob('workflow.knime'):
        return item.parent
    
    raise FileNotFoundError("No workflow.knime found in archive")


def main():
    parser = argparse.ArgumentParser(
        description="Convert KNIME workflow (.knwf) to Python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transpile.py meu_fluxo.knwf
  python transpile.py meu_fluxo.knwf --output pipeline.py
  python transpile.py meu_fluxo.knwf --no-llm --verbose
        """
    )
    
    parser.add_argument("knwf_file", help="Path to KNIME workflow file (.knwf)")
    parser.add_argument("-o", "--output", help="Output Python file (default: same name as input)")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM fallback (faster)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--report", help="Generate markdown report (default: <output>_report.md)")
    
    args = parser.parse_args()
    
    # Validate input file
    knwf_path = Path(args.knwf_file).resolve()
    if not knwf_path.exists():
        print(f"❌ Error: File not found: {knwf_path}")
        sys.exit(1)
    
    if not knwf_path.suffix.lower() == '.knwf':
        print(f"⚠️ Warning: File is not .knwf: {knwf_path}")
    
    # Determine output paths - save next to input file
    input_dir = knwf_path.parent
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = input_dir / f"{knwf_path.stem}.py"
    
    report_path = Path(args.report) if args.report else input_dir / f"{knwf_path.stem}_report.md"
    
    print("="*60)
    print("🔄 ChatKnime Transpiler")
    print("="*60)
    print(f"Input:  {knwf_path}")
    print(f"Output: {output_path}")
    print(f"LLM:    {'Disabled' if args.no_llm else 'Enabled'}")
    print("="*60)
    
    try:
        from tests.standalone_transpiler import StandaloneTranspiler
        
        # Check for pre-analyzed JSON
        analysis_path = knwf_path.parent / "workflow_analysis_v2.json"
        
        if not analysis_path.exists():
            # Look in ChatKnime root
            analysis_path = Path(__file__).parent.parent / "workflow_analysis_v2.json"
        
        if analysis_path.exists():
            print(f"\n📂 Using pre-analyzed data: {analysis_path.name}")
            transpiler = StandaloneTranspiler(use_llm=not args.no_llm)
            analysis = transpiler.load_analysis(analysis_path)
            code = transpiler.transpile(analysis)
            
            # Save output
            output_path.write_text(code, encoding='utf-8')
            
            # Generate report
            report_content = f"""# Transpilation Report

## Summary
- **Input**: {knwf_path.name}
- **Output**: {output_path.name}
- **Total Nodes**: {transpiler.stats['total']}
- **Coverage**: {100.0 if transpiler.stats['total'] > 0 else 0:.1f}%

## Statistics
| Type | Count |
|------|-------|
| Template | {transpiler.stats['template']} |
| LLM | {transpiler.stats['llm']} |
| Pattern | {transpiler.stats.get('pattern', 0)} |
| Fallback | {transpiler.stats['fallback']} |
"""
            report_path.write_text(report_content, encoding='utf-8')
            
            print("\n" + "="*60)
            print("✅ TRANSPILATION COMPLETE!")
            print("="*60)
            print(f"Python file: {output_path}")
            print(f"Report:      {report_path}")
            print(f"Nodes:       {transpiler.stats['total']}")
            print(f"Coverage:    100.0%")
            print("="*60)
        else:
            print(f"\n❌ Pre-analyzed data not found!")
            print(f"   Expected: {analysis_path}")
            print("\nPlease run the workflow analyzer first, or use e2e_complete_transpilation.py")
            sys.exit(1)
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the backend directory with venv activated.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

