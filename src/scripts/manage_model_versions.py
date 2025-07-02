import os
import json
import logging
from pathlib import Path
import shutil
from datetime import datetime
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelVersionManager:
    def __init__(self, 
                model_dir: str = "models/ensemble",
                version_file: str = "models/versions.json"):
        self.model_dir = Path(model_dir)
        self.version_file = Path(version_file)
        self.current_version = None
        self.versions = {}
        
    def load_versions(self) -> None:
        """Load model versions from file."""
        if self.version_file.exists():
            with open(self.version_file, 'r') as f:
                self.versions = json.load(f)
                self.current_version = self.versions.get('current_version')
        else:
            self.versions = {
                'current_version': None,
                'versions': {}
            }
            
    def save_versions(self) -> None:
        """Save model versions to file."""
        with open(self.version_file, 'w') as f:
            json.dump(self.versions, f, indent=4)
            
    def create_new_version(self, 
                         model_path: str,
                         model_type: str = "lstm",
                         metrics: Optional[Dict[str, float]] = None) -> str:
        """Create a new model version."""
        # Generate version number
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v{timestamp}"
        
        # Create version directory
        version_dir = self.model_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        if model_type == "lstm":
            shutil.copytree(model_path, version_dir)
        elif model_type == "transformer":
            shutil.copytree(model_path, version_dir)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Update versions
        self.load_versions()
        self.versions['versions'][version] = {
            'timestamp': timestamp,
            'model_type': model_type,
            'metrics': metrics or {},
            'path': str(version_dir)
        }
        self.versions['current_version'] = version
        self.current_version = version
        
        self.save_versions()
        
        logger.info(f"Created new model version: {version}")
        return version
        
    def get_version_info(self, version: str) -> Dict:
        """Get information about a specific version."""
        self.load_versions()
        return self.versions['versions'].get(version, {})
        
    def list_versions(self) -> List[Dict]:
        """List all available versions."""
        self.load_versions()
        return [
            {
                'version': v,
                'timestamp': info['timestamp'],
                'model_type': info['model_type'],
                'metrics': info['metrics']
            }
            for v, info in self.versions['versions'].items()
        ]
        
    def rollback_to_version(self, version: str) -> None:
        """Rollback to a specific version."""
        version_info = self.get_version_info(version)
        if not version_info:
            raise ValueError(f"Version {version} not found")
            
        # Update current version
        self.load_versions()
        self.versions['current_version'] = version
        self.current_version = version
        
        # Update deployment
        deployer = ModelDeployer()
        deployer.prepare_model_for_serving()
        deployer.start_serving()
        
        self.save_versions()
        logger.info(f"Rolled back to version: {version}")
        
    def compare_versions(self, version1: str, version2: str) -> Dict:
        """Compare two model versions."""
        info1 = self.get_version_info(version1)
        info2 = self.get_version_info(version2)
        
        if not info1 or not info2:
            raise ValueError("One or both versions not found")
            
        # Compare metrics
        metrics1 = info1.get('metrics', {})
        metrics2 = info2.get('metrics', {})
        
        comparison = {}
        for metric in set(metrics1.keys()).union(metrics2.keys()):
            val1 = metrics1.get(metric, None)
            val2 = metrics2.get(metric, None)
            
            if val1 is not None and val2 is not None:
                comparison[metric] = {
                    version1: val1,
                    version2: val2,
                    'difference': val2 - val1
                }
        
        return comparison
        
    def delete_version(self, version: str) -> None:
        """Delete a specific version."""
        version_info = self.get_version_info(version)
        if not version_info:
            raise ValueError(f"Version {version} not found")
            
        # Delete version directory
        version_dir = Path(version_info['path'])
        if version_dir.exists():
            shutil.rmtree(version_dir)
            
        # Update versions
        self.load_versions()
        del self.versions['versions'][version]
        
        if self.current_version == version:
            # Set current version to latest if deleted version was current
            latest_version = max(self.versions['versions'].keys())
            self.versions['current_version'] = latest_version
            self.current_version = latest_version
            
        self.save_versions()
        logger.info(f"Deleted version: {version}")
        
    def promote_version(self, version: str) -> None:
        """Promote a version to production."""
        version_info = self.get_version_info(version)
        if not version_info:
            raise ValueError(f"Version {version} not found")
            
        # Update current version
        self.load_versions()
        self.versions['current_version'] = version
        self.current_version = version
        
        # Update deployment
        deployer = ModelDeployer()
        deployer.prepare_model_for_serving()
        deployer.start_serving()
        
        self.save_versions()
        logger.info(f"Promoted version {version} to production")

def main():
    """Main function to manage model versions."""
    manager = ModelVersionManager()
    
    # Example usage
    # Create new version
    new_version = manager.create_new_version(
        model_path="models/lstm_model.h5",
        model_type="lstm",
        metrics={'soh_mae': 0.05, 'soc_mae': 0.04}
    )
    
    # List all versions
    versions = manager.list_versions()
    print("Available versions:", versions)
    
    # Compare versions
    if len(versions) > 1:
        comparison = manager.compare_versions(
            versions[0]['version'],
            versions[1]['version']
        )
        print("Version comparison:", comparison)
    
    # Promote version
    manager.promote_version(new_version)

if __name__ == "__main__":
    main()
