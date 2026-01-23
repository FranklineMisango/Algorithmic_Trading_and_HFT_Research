"""
Export trained ML model for QuantConnect Lean integration

This script exports the trained model from the notebook environment
to the Lean directory in a format compatible with QuantConnect.

Usage:
    python export_model_for_lean.py --model-path ../models/xgboost_best.pkl
"""

import argparse
import joblib
import pickle
import json
from pathlib import Path


def export_model(model_path, output_dir='lean'):
    """
    Export model and feature names for Lean integration
    
    Args:
        model_path: Path to the trained model pickle file
        output_dir: Output directory (default: 'lean')
    """
    print(f"Exporting model from: {model_path}")
    
    # Load model
    try:
        model_data = joblib.load(model_path)
        print(f"✓ Loaded model from {model_path}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False
    
    # Extract model and metadata
    if isinstance(model_data, dict):
        model = model_data.get('model')
        feature_names = model_data.get('feature_names', [])
        model_type = model_data.get('model_type', 'unknown')
    else:
        model = model_data
        feature_names = []
        model_type = type(model).__name__
    
    print(f"  Model type: {model_type}")
    print(f"  Features: {len(feature_names)}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save model
    model_file = output_path / 'trained_model.pkl'
    joblib.dump(model, model_file)
    print(f"✓ Model saved to: {model_file}")
    
    # Save feature names
    if feature_names:
        feature_file = output_path / 'feature_names.txt'
        with open(feature_file, 'w') as f:
            f.write('\n'.join(feature_names))
        print(f"✓ Feature names saved to: {feature_file}")
    
    # Save metadata
    metadata = {
        'model_type': model_type,
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'source_model': str(model_path)
    }
    
    metadata_file = output_path / 'model_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {metadata_file}")
    
    print("\n" + "="*60)
    print("Model export complete!")
    print("="*60)
    print("\nNext steps:")
    print(f"1. Verify files in {output_dir}/ directory")
    print("2. Review lean/main.py for model integration")
    print("3. Run backtest: lean backtest \"lean\"")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Export trained ML model for QuantConnect Lean'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model file (e.g., models/xgboost_best.pkl)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='lean',
        help='Output directory (default: lean)'
    )
    
    args = parser.parse_args()
    
    export_model(args.model_path, args.output_dir)


if __name__ == '__main__':
    main()
