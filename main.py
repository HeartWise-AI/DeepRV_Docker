import os
import sys
import yaml
import json
import numpy as np
import pandas as pd
from pprint import pprint
from utils.parser import HearWiseArgs
from utils.constants import MODEL_MAPPING

from heartwise_statplots.utils import HuggingFaceWrapper
from heartwise_statplots.utils.api import load_api_keys
from heartwise_statplots.metrics import MetricsComputer, ClassificationMetrics


# Add the parent directory to the system path
sys.path.append('./Orion')
from orion.utils.video_training_and_eval import perform_inference


def get_model_weights(
    model_name: str, 
    hugging_face_api_key: str
)->str:
    """
    Retrieves the local path to the specified model's weights from Hugging Face.

    This function uses the `HuggingFaceWrapper` to download the model weights
    from the Hugging Face repository and stores them in a designated local directory.

    Args:
        model_name (str): The name of the model to retrieve weights for.
        hugging_face_api_key (str): The API key for authenticating with Hugging Face.

    Returns:
        str: The local file system path where the model weights are stored.
    """    
    model_path = HuggingFaceWrapper.get_model(
        repo_id=f"heartwise/{model_name}",
        local_dir=os.path.join("weights", model_name),
        hugging_face_api_key=hugging_face_api_key
    )
    return model_path


def setup_orion_config(
    args: HearWiseArgs, 
    default_model_config: dict[str, str]
)->dict:
    with open(args.config_path) as file:
        config = yaml.safe_load(file)
    config['output_dir'] = args.output_folder
    config['model_path'] = args.model_path
    config['data_filename'] = args.data_path
    config['batch_size'] = args.batch_size
    config['num_workers'] = args.num_workers
    config['device'] = args.model_device
    config.update(default_model_config)
    return config


def compute_metrics(df_predictions_inference: pd.DataFrame)->dict:
    """
    Computes classification metrics (AUC, AUPRC, F1 Score) based on predictions and true labels.

    Args:
        df_predictions_inference (pd.DataFrame): DataFrame containing 'y_hat' for predictions and 'y_true' for true labels.

    Returns:
        dict: A dictionary containing the computed metrics.
    """    
    y_pred = df_predictions_inference['y_hat'].to_numpy().astype(np.float64)
    y_true = df_predictions_inference['y_true'].to_numpy().astype(np.int64)
    metrics = MetricsComputer.compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        metrics=[
            ClassificationMetrics.AUC, 
            ClassificationMetrics.AUPRC, 
            ClassificationMetrics.F1_SCORE
        ],
        bootstrap=True,
        n_iterations=1000
    )
    return metrics

def main(args: HearWiseArgs)->None:
    models = {}
    # Get model weights
    if args.use_x3d:
        hugging_face_model_name = MODEL_MAPPING['x3d']['hugging_face_model_name']
        config = MODEL_MAPPING['x3d']['config']
        models[hugging_face_model_name] = config
        print(f"Using {hugging_face_model_name} model")
    if args.use_mvit:
        hugging_face_model_name = MODEL_MAPPING['mvit']['hugging_face_model_name']
        config = MODEL_MAPPING['mvit']['config']
        models[hugging_face_model_name] = config
        print(f"Using {hugging_face_model_name} model")
    
    # Load API key
    hugging_face_api_key = load_api_keys(args.hugging_face_api_key_path)['HUGGING_FACE_API_KEY']
    
    # Run evaluation pipeline for each model
    for model in models:
        print(f"Running evaluation for {model} model")
        
        # Get model weights
        model_weights_path = get_model_weights(model, hugging_face_api_key)
        pt_file = next((f for f in os.listdir(model_weights_path) if f.endswith('.pt')), None)
        if not pt_file:
            raise ValueError("No .pt file found in the directory")
        args.model_path = os.path.join(model_weights_path, pt_file)    
        
        # Setup orion config
        config = setup_orion_config(args, models[model])
        pprint(config)
        
        # Run inference
        df_predictions_inference = perform_inference(config=config, split='inference', log_wandb=False)
        
        # Compute metrics
        metrics = compute_metrics(df_predictions_inference)
    
        model_metrics = {
            model: metrics
        }
        
        pprint(model_metrics)
        
        with open(os.path.join(args.output_folder, f"{model}_metrics.json"), "w") as f:
            json.dump(model_metrics, f, indent=4)
    
    
if __name__ == "__main__":
    args = HearWiseArgs.parse_arguments()
    print("Summary of the arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    main(args)




