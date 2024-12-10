import os
import cv2
import sys
import yaml
import json
import shutil
import logging
import pydicom
import numpy as np
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
from pathlib import Path
from typing import Optional

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


def normalize_pixel_array(pixel_array: np.ndarray)->np.ndarray:
    """
    Normalize the pixel array to the 0-255 range and convert to uint8.
    """
    pixel_min = np.min(pixel_array)
    pixel_max = np.max(pixel_array)
    
    if pixel_max == pixel_min:
        # Avoid division by zero; return a zero array
        return np.zeros(pixel_array.shape, dtype=np.uint8)

    # Normalize to 0-255
    normalized = (pixel_array - pixel_min) / (pixel_max - pixel_min)
    normalized = (normalized * 255).astype(np.uint8)
    return normalized


def convert_dicom_to_avi(
    input_path: str, 
    output_path: str
)->Optional[str]:
    """
    Converts DICOM videos to AVI format.
    
    Args:
        input_path (str): The path to the DICOM file.
    
    Returns:
        str: The path to the AVI file.
    """
    ds = pydicom.dcmread(input_path)
    
    # Create output filename
    output_filename = Path(input_path).stem + '.avi'
    output_path = Path(output_path) / output_filename
    
    # Extract FPS; ensure the DICOM tag exists
    frame_rate_tag = (0x08, 0x2144)  # This tag may vary depending on the DICOM file
    if frame_rate_tag in ds:
        fps = float(ds[frame_rate_tag].value)
    else:
        fps = 30.0  # Default FPS if not specified
        
    try:
        photometrics = ds.PhotometricInterpretation
        if photometrics not in ['MONOCHROME1', 'MONOCHROME2', 'RGB']:
            print(ValueError(f"Unsupported Photometric Interpretation: {photometrics} - with shape{ds.pixel_array.shape}"))
            return
    except:
        print(f"Error in reading {input_path}")
        return
    
    conversion_fn = cv2.COLOR_GRAY2BGR if photometrics == 'MONOCHROME1' or photometrics == 'MONOCHROME2' else cv2.COLOR_RGB2BGR

    fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, ds.pixel_array.shape[1:3])
    
    for frame in ds.pixel_array:
        # frame = normalize_pixel_array(frame) TODO: No pixel array normalization for now.
        frame = cv2.cvtColor(frame, conversion_fn)
        out.write(frame)
    
    out.release()
    return str(output_path)


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


def process_dicom_batch(dicom_batch):
    """
    Process a batch of DICOM files in parallel.
    
    Args:
        dicom_batch (tuple): Tuple containing (dicom_filepath, output_path)
    
    Returns:
        tuple: (original_path, new_path) or (original_path, None) if conversion failed
    """
    dicom_filepath, output_path = dicom_batch
    try:
        avi_filepath = convert_dicom_to_avi(input_path=dicom_filepath, output_path=output_path)
        return (dicom_filepath, avi_filepath)
    except Exception as e:
        logger.error(f"Error converting {dicom_filepath}: {str(e)}")
        return (dicom_filepath, None)


def main(args: HearWiseArgs)->None:   
    try:     
        # Define tmp dir
        tmp_dir = Path('/app/tmp')
        tmp_dir.mkdir(parents=True, exist_ok=True)
        
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
        # Sort input dataframe by 'FileName'
        input_df_sorted = input_df.sort_values(by='FileName')
        

        # Convert DICOM videos to AVI using multiprocessing
        logging.info("Check for DICOM files and convert to AVI")
        dicom_filepaths = input_df_sorted[input_df_sorted['FileName'].apply(lambda x: x.endswith('.dcm'))]['FileName'].tolist()
        logger.info(f"{len(dicom_filepaths)} DICOM files found in input dataframe.")
        
        if len(dicom_filepaths) > 0:
            # Prepare batches for parallel processing
            num_processes = min(args.preprocessing_workers, mp.cpu_count())  # Use available CPU cores
            print(f"Number of processes: {num_processes}")
            dicom_batches = [(filepath, tmp_dir) for filepath in dicom_filepaths]
            
            converted_count = 0
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = [executor.submit(process_dicom_batch, batch) for batch in dicom_batches]
                
                # Process results as they complete
                for future in tqdm(as_completed(futures), total=len(dicom_filepaths), desc="Converting DICOM to AVI"):
                    original_path, new_path = future.result()
                    if new_path:
                        input_df_sorted.loc[input_df_sorted['FileName'] == original_path, 'FileName'] = new_path
                        converted_count += 1
            
            logger.info(f"Converted {converted_count} DICOM files to AVI format in {tmp_dir}")
            
        # save new input dataframe
        input_df_sorted.to_csv(tmp_dir / "input_df_sorted.csv", index=False, sep='α')
        args.data_path = tmp_dir / "input_df_sorted.csv"
            
            
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
        predicted_views_path = tmp_dir / "predicted_views.csv"
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
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            logger.error(f"Failed to write metrics to {output_metrics_path}: {e}")
            raise
        
        # Remove tmp dir
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
  
    except Exception as e:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        logger.exception(f"An error occurred during execution: {e}")
        sys.exit(1)       
    
    
if __name__ == "__main__":
    args = HearWiseArgs.parse_arguments()
    print("Summary of the arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    main(args)