import tensorflow as tf
# import tensorflow_model_server as tfs
import numpy as np
import logging
from pathlib import Path
import json
import requests
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelDeployer:
    def __init__(self, model_dir: str = "models/ensemble",
                serving_port: int = 8501,
                model_name: str = "battery_prediction"):
        self.model_dir = Path(model_dir)
        self.serving_port = serving_port
        self.model_name = model_name
        self.server = None

    def prepare_model_for_serving(self) -> None:
        """Prepare the ensemble model for TensorFlow Serving."""
        # Load ensemble model components
        lstm_model = tf.keras.models.load_model(str(self.model_dir / 'lstm_model.h5'))

        # Create serving model directory structure
        serving_dir = self.model_dir / 'serving'
        version_dir = serving_dir / '1'
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save model in SavedModel format
        tf.saved_model.save(
            lstm_model,
            str(version_dir),
            signatures={
                'serving_default': lstm_model.serve.get_concrete_function(
                    tf.TensorSpec(shape=[None, None, lstm_model.input_shape[-1]],
                                dtype=tf.float32,
                                name='input')
                )
            }
        )

        # Save model metadata
        metadata = {
            'model_name': self.model_name,
            'version': '1.0.0',
            'input_shape': lstm_model.input_shape,
            'output_shape': lstm_model.output_shape
        }

        with open(serving_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)

        logger.info("Model prepared for serving")

    def start_serving(self) -> None:
        """Start TensorFlow Serving server."""
        pass
        # try:
        #     # Start TensorFlow Serving
        #     self.server = tfs.TFModelServer(
        #         port=self.serving_port,
        #         model_name=self.model_name,
        #         model_path=str(self.model_dir / 'serving')
        #     )
        #     self.server.start()
        #     logger.info(f"TensorFlow Serving started on port {self.serving_port}")
        # except Exception as e:
        #     logger.error(f"Failed to start TensorFlow Serving: {str(e)}")
        #     raise

    def make_prediction(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Make prediction using the deployed model."""
        # Prepare request
        data = data.astype(np.float32)

        # Send request to TensorFlow Serving
        data_json = json.dumps({
            "signature_name": "serving_default",
            "instances": data.tolist()
        })

        headers = {"content-type": "application/json"}
        json_response = requests.post(
            f'http://localhost:{self.serving_port}/v1/models/{self.model_name}:predict',
            data=data_json,
            headers=headers
        )

        if json_response.status_code != 200:
            raise Exception(f"Prediction failed: {json_response.text}")

        # Parse response
        response = json_response.json()
        predictions = response['predictions'][0]

        return {
            'soh_prediction': predictions['soh_output'],
            'soc_prediction': predictions['soc_output']
        }

    def create_dockerfile(self) -> None:
        """Create Dockerfile for model deployment."""
        dockerfile_content = f"""
FROM tensorflow/serving

# Copy model files
COPY {self.model_dir.name}/serving /models/{self.model_name}

# Set environment variables
ENV MODEL_NAME={self.model_name}
ENV MODEL_BASE_PATH=/models

# Start TensorFlow Serving
CMD ["tensorflow_model_server", \\
    "--port=8501", \\
    "--rest_api_port=8501", \\
    "--model_name=${{MODEL_NAME}}", \\
    "--model_base_path=${{MODEL_BASE_PATH}}/${{MODEL_NAME}}"]
"""

        with open('Dockerfile.serving', 'w') as f:
            f.write(dockerfile_content)

        logger.info("Dockerfile created for model deployment")

    def create_api_server(self) -> None:
        """Create a REST API server for the model."""
        api_content = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import numpy as np
import json
from typing import List

app = FastAPI()

class PredictionInput(BaseModel):
    data: List[List[List[float]]]

class PredictionOutput(BaseModel):
    soh_prediction: float
    soc_prediction: float

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        # Convert input data to numpy array
        data = np.array(input_data.data, dtype=np.float32)

        # Prepare request
        data_json = json.dumps({
            "signature_name": "serving_default",
            "instances": data.tolist()
        })

        # Send request to TensorFlow Serving
        headers = {"content-type": "application/json"}
        response = requests.post(
            "http://tensorflow_serving:8501/v1/models/battery_prediction:predict",
            data=data_json,
            headers=headers
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Prediction failed")

        # Parse response
        predictions = response.json()['predictions'][0]

        return PredictionOutput(
            soh_prediction=float(predictions['soh_output']),
            soc_prediction=float(predictions['soc_output'])
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
'''
        with open('api_server.py', 'w') as f:
            f.write(api_content)

        logger.info("API server created")

def main():
    """Main function to deploy model."""
    # Initialize deployer
    deployer = ModelDeployer()

    # Prepare model for serving
    deployer.prepare_model_for_serving()

    # Start serving
    # deployer.start_serving() # This will block, so comment out for script execution

    logger.info("Model deployment completed successfully")

if __name__ == "__main__":
    main()
