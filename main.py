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
            ClassificationMetrics.SENSITIVITY,
            ClassificationMetrics.SPECIFICITY,
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
    
    input_df = pd.read_csv(args.data_path, sep='α')
    if 'examen' in args.eval_granularity and 'Examen_ID' not in input_df.columns:
        raise ValueError("Examen_ID column is required for examen granularity evaluation")
    
    # Initialize model metrics dictionary
    model_metrics = {
        model: {
            granularity: {} for granularity in args.eval_granularity
        } for model in models
    }
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
        for granularity in args.eval_granularity:
            print(f"Computing {granularity} metrics for {model} model")
            if granularity == 'examen':
                # Sort by 'FileName' in alphabetical order
                df_predictions_inference.rename(columns={'filename': 'FileName'}, inplace=True)
                df_predictions_inference_sorted = df_predictions_inference.sort_values(by='FileName')
                input_df_sorted = input_df.sort_values(by='FileName')
                
                merged_df = pd.merge(input_df_sorted, df_predictions_inference_sorted, on='FileName')
                
                # Compute mean logits based on 'Examen_ID'
                df_grouped = merged_df.groupby('Examen_ID').agg({
                    'y_hat': 'mean',
                    'y_true': 'first'  # Assuming y_true is the same for the same Examen_ID
                }).reset_index()
    
                # Use the grouped DataFrame for metric computation                
                metrics = compute_metrics(df_grouped)
                
            elif granularity == 'video':
                metrics = compute_metrics(df_predictions_inference)
        
            # Store metrics
            model_metrics[model][granularity] = metrics
        
        pprint(model_metrics)
        
    # Save metrics to JSON file
    with open(os.path.join(args.output_folder, f"deeprv_evaluation_metrics.json"), "w") as f:
        json.dump(model_metrics, f, indent=4)
    
    
if __name__ == "__main__":
    args = HearWiseArgs.parse_arguments()
    print("Summary of the arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    main(args)




