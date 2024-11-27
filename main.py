import os
import sys
import yaml
import json
import numpy as np
import pandas as pd
from utils.parser import HearWiseArgs
from utils.constants import MODEL_MAPPING

from heartwise_statplots.utils import HuggingFaceWrapper
from heartwise_statplots.utils.api import load_api_keys
from heartwise_statplots.metrics import MetricsComputer, ClassificationMetrics


# Add the parent directory to the system path
sys.path.append('./Orion')
from orion.utils.video_training_and_eval import perform_inference


def get_model_weights(model_name: str, hugging_face_api_key: str)->str:
    model_path = HuggingFaceWrapper.get_model(
        repo_id=f"heartwise/{model_name}",
        local_dir=os.path.join("weights", model_name),
        hugging_face_api_key=hugging_face_api_key
    )
    return model_path


def run_inference(args: HearWiseArgs)->pd.DataFrame:   
    
    with open(args.config_path) as file:
        config = yaml.safe_load(file)
    config['output_dir'] = args.output_folder
    config['model_path'] = args.model_path
    config['data_filename'] = args.data_path
    print(config)
    return perform_inference(config=config, split='inference', log_wandb=False)
    

def main(args: HearWiseArgs)->None:
    # Get model weights
    if args.use_x3d:
        model_name = MODEL_MAPPING['x3d']
    elif args.use_mvit:
        model_name = MODEL_MAPPING['mvit']
    else:
        raise ValueError("Invalid model name")
    
    # Load API key
    hugging_face_api_key = load_api_keys(args.hugging_face_api_key_path)['HUGGING_FACE_API_KEY']
    
    # Get model weights
    model_weights_path = get_model_weights(model_name, hugging_face_api_key)
    pt_file = next((f for f in os.listdir(model_weights_path) if f.endswith('.pt')), None)
    if not pt_file:
        raise ValueError("No .pt file found in the directory")
    args.model_path = os.path.join(model_weights_path, pt_file)    
        
    # Run inference
    df_predictions_inference = run_inference(args)
    print(df_predictions_inference.columns.tolist())
    
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
   
    model_metrics = {
        model_name: metrics
    }
    
    print(model_metrics)
    
    with open(os.path.join(args.output_folder, "model_metrics.json"), "w") as f:
        json.dump(model_metrics, f, indent=4)
    
    
if __name__ == "__main__":
    args = HearWiseArgs.parse_arguments()
    print("Summary of the arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    main(args)




