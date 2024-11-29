import os
import sys
import yaml
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from utils.parser import HearWiseArgs
from utils.constants import (
    MODEL_MAPPING,
    AngioClasses
)

from heartwise_statplots.utils import HuggingFaceWrapper
from heartwise_statplots.utils.api import load_api_keys
from heartwise_statplots.metrics import MetricsComputer, ClassificationMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)

# Add the parent directory to the system path using pathlib
orion_path = Path(__file__).parent / 'Orion'
sys.path.append(str(orion_path))
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
    return HuggingFaceWrapper.get_model(
        repo_id=f"heartwise/{model_name}",
        local_dir=os.path.join("weights", model_name),
        hugging_face_api_key=hugging_face_api_key
    )


def setup_orion_config(
    args: HearWiseArgs, 
    default_model_config: dict[str, str]
)->dict:
    """
    Sets up the Orion configuration for a given model.

    Args:
        args (HearWiseArgs): The command-line arguments.
        default_model_config (dict): The default model configuration.

    Returns:
        dict: The updated Orion configuration.
    """
    try:
        with open(args.config_path) as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {args.config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise            
    config['output_dir'] = args.output_folder
    config['model_path'] = args.model_path
    config['data_filename'] = args.data_path
    config['batch_size'] = args.batch_size
    config['num_workers'] = args.num_workers
    config['device'] = args.model_device
    config.update(default_model_config)
    return config

def get_model_path(
    model: str, 
    hugging_face_api_key: str
)->str:
    """
    Retrieves the local path to the specified model's weights from Hugging Face.

    Args:
        model (str): The name of the model to retrieve weights for.
        hugging_face_api_key (str): The API key for authenticating with Hugging Face.

    Returns:
        str: The local file system path where the model weights are stored.
    """
    # Get model weights
    model_weights_path = get_model_weights(model, hugging_face_api_key)
    pt_file = next((f for f in os.listdir(model_weights_path) if f.endswith('.pt')), None)
    if not pt_file:
        raise ValueError("No .pt file found in the directory")    
    return os.path.join(model_weights_path, pt_file)


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
    try:     
        # Load API key
        api_keys = load_api_keys(args.hugging_face_api_key_path)
        hugging_face_api_key = api_keys.get('HUGGING_FACE_API_KEY')
        if not hugging_face_api_key:
            logger.error("HUGGING_FACE_API_KEY not found in the provided API keys file.")
            raise KeyError("HUGGING_FACE_API_KEY not found.")
    
    
        # Read input dataframe
        input_path = Path(args.data_path)
        if not input_path.exists():
            logger.error(f"Data file not found at: {input_path}")
            raise FileNotFoundError(f"Data file not found at: {input_path}")

        input_df = pd.read_csv(input_path, sep='α')
        if 'examen' in args.eval_granularity and 'Examen_ID' not in input_df.columns:
            logger.error("Examen_ID column is missing from the input data.")
            raise KeyError("Examen_ID column is required for examen granularity evaluation.")
    
        input_df_sorted = input_df.sort_values(by='FileName')
        
            
        # Initialize DeepRV models
        deeprv_models = {}        
        if args.use_x3d:
            hugging_face_model_name = MODEL_MAPPING['x3d']['hugging_face_model_name']
            config = MODEL_MAPPING['x3d']['config']
            deeprv_models[hugging_face_model_name] = config
            logger.info(f"Using {hugging_face_model_name} model")
        if args.use_mvit:
            hugging_face_model_name = MODEL_MAPPING['mvit']['hugging_face_model_name']
            config = MODEL_MAPPING['mvit']['config']
            deeprv_models[hugging_face_model_name] = config
            logger.info(f"Using {hugging_face_model_name} model")


        # Initialize Angio classifier
        angio_classifier_hugging_face_model_name = MODEL_MAPPING['swin3d_s_angio_video_classifier']['hugging_face_model_name']
        angio_classifier_config = MODEL_MAPPING['swin3d_s_angio_video_classifier']['config']
        logger.info(f"Using {angio_classifier_hugging_face_model_name} model")
        args.model_path = get_model_path(
            model=angio_classifier_hugging_face_model_name, 
            hugging_face_api_key=hugging_face_api_key
        )
        
        # Setup orion config for Angio classifier
        angio_updated_config = setup_orion_config(
            args=args, 
            default_model_config=angio_classifier_config
        )    
        logger.info(f"Angio updated config: {angio_updated_config}")
        
        # Run inference for Angio classifier
        df_predicted_views = perform_inference(
            config=angio_updated_config, 
            split='inference', 
            log_wandb=False
        )
        df_predicted_views.rename(columns={'filename': 'FileName'}, inplace=True)
        df_predicted_views_sorted = df_predicted_views.sort_values(by='FileName')
        count_lad = (df_predicted_views_sorted['argmax_class'] == AngioClasses.LAD.value).sum()
        count_mid_rca = (df_predicted_views_sorted['argmax_class'] == AngioClasses.MID_RCA.value).sum()
        logger.info(f"LAD count: {count_lad}")
        logger.info(f"MID_RCA count: {count_mid_rca}")
        
        # Filter predicted views for LAD or MID_RCA
        df_predicted_views_sorted = df_predicted_views_sorted[
            (df_predicted_views_sorted['argmax_class'] == AngioClasses.LAD.value) |
            (df_predicted_views_sorted['argmax_class'] == AngioClasses.MID_RCA.value)
        ]
        logger.info(f"Predicted views count after filtering: {len(df_predicted_views_sorted)}")
        
        # Filter input dataframe for predicted views
        input_df_sorted = input_df_sorted[input_df_sorted['FileName'].isin(df_predicted_views_sorted['FileName'])]
 
        # Check if the number of predicted views matches the number of input views
        total = count_lad + count_mid_rca
        if total != len(df_predicted_views_sorted):
            logger.error(f"Total predicted views ({total}) do not match the number of predicted views ({len(df_predicted_views_sorted)})")
            raise ValueError(f"Total predicted views ({total}) do not match the number of predicted views ({len(df_predicted_views_sorted)})")
        if total != len(input_df_sorted):
            logger.error(f"Total predicted views ({total}) do not match the number of input views ({len(input_df_sorted)})")
            raise ValueError(f"Total predicted views ({total}) do not match the number of input views ({len(input_df_sorted)})")
        
        # Save filtered input dataframe to csv for DeepRV and Coronary Dominance classifier
        predicted_views_path = Path(input_path).parent / "predicted_views.csv"
        input_df_sorted.to_csv(predicted_views_path, index=False, sep='α')
        args.data_path = predicted_views_path
              
              
        # Initialize Coronary Dominance classifier
        coronary_classifier_hugging_face_model_name = MODEL_MAPPING['swin3d_s_coronary_dominance_classifier']['hugging_face_model_name']
        coronary_classifier_config = MODEL_MAPPING['swin3d_s_coronary_dominance_classifier']['config']
        logger.info(f"Using {coronary_classifier_hugging_face_model_name} model")
        args.model_path = get_model_path(
            model=coronary_classifier_hugging_face_model_name, 
            hugging_face_api_key=hugging_face_api_key
        )
        
        # Setup orion config for Coronary Dominance classifier
        coronary_updated_config = setup_orion_config(
            args=args, 
            default_model_config=coronary_classifier_config
        )
        logger.info(f"Coronary updated config: {coronary_updated_config}")

        # Run inference for Coronary Dominance classifier
        df_predicted_views = perform_inference(
            config=coronary_updated_config, 
            split='inference', 
            log_wandb=False
        )


        # Initialize model metrics dictionary
        model_metrics = {
            model: {
                granularity: {} for granularity in args.eval_granularity
            } for model in deeprv_models
        }
         
        # Run evaluation pipeline for each model
        for model in deeprv_models:
            logger.info(f"Running evaluation for {model} model")
            args.model_path = get_model_path(
                model=model, 
                hugging_face_api_key=hugging_face_api_key
            )
            
            # Setup orion config
            config = setup_orion_config(
                args=args, 
                default_model_config=deeprv_models[model]
            )
            logger.info(f"DeepRV updated config: {config}")
            
            # Run inference
            df_predictions_inference = perform_inference(
                config=config, 
                split='inference', 
                log_wandb=False
            )
            
            # Compute metrics
            for granularity in args.eval_granularity:
                logger.info(f"Computing {granularity} metrics for {model} model")
                if granularity == 'examen':
                    # Sort by 'FileName' in alphabetical order
                    df_predictions_inference.rename(columns={'filename': 'FileName'}, inplace=True)
                    df_predictions_inference_sorted = df_predictions_inference.sort_values(by='FileName')
                    
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
            
            logger.info(f"Model metrics: {model_metrics}")
            
            
        # Save metrics to JSON file
        output_metrics_path = Path(args.output_folder) / "deeprv_evaluation_metrics.json"
        try:
            with open(output_metrics_path, "w") as f:
                json.dump(model_metrics, f, indent=4)
            logger.info(f"Metrics successfully saved to {output_metrics_path}")
        except IOError as e:
            logger.error(f"Failed to write metrics to {output_metrics_path}: {e}")
            raise
  
    except Exception as e:
        logger.exception(f"An error occurred during execution: {e}")
        sys.exit(1)       
    
    
if __name__ == "__main__":
    args = HearWiseArgs.parse_arguments()
    print("Summary of the arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    main(args)