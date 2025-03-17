# DeepRV_Docker

## Table of Contents
- 🚀 [Features](#features)
- 🛠️ [Installation](#installation)
- 📄 [Usage](#usage)
- 🐳 [Docker](#docker)
- 🤝 [Contributing](#contributing)
- 📚 [Citation](#citation)

## 🚀 Features

- 1. X3D-based multilabel classification model for video (RV function binary classification) 
   Request access to the model on [HuggingFace](https://huggingface.co/heartwise/DeepRV_x3d) 🤖
- 2. MViT-based multilabel classification model for video (RV function binary classification) 
   Request access to the model on [HuggingFace](https://huggingface.co/heartwise/DeepRV_mvit) 🤖
- 3. Swin3D-based binary classification model for video (Angio classification) 
   Request access to the model on [HuggingFace](https://huggingface.co/heartwise/DeepRV_swin3d_s_angio_video_classifier) 🤖
- 4. Swin3D-based binary classification model for video (Coronary Dominance classification)
   Request access to the model on [HuggingFace](https://huggingface.co/heartwise/DeepRV_swin3d_s_coronary_dominance) 🤖
- Generate examen-based and video-based metrics (AUC, Accuracy, Sensitivity, Specificity, F1-score)
- Dockerized deployment for easy setup and execution
- Configurable pipeline for flexible usage
- CPU & GPU support for accelerated processing

## 🛠️ Installation 

1. 📥 Clone the repository:
   ```
   git clone https://github.com/HeartWise-AI/DeepRV_Docker.git
   cd DeepRV_Docker
   ```

2. 🔑 Set up your HuggingFace API key:
   - Create a HuggingFace account if you don't have one yet
   - Ask for access to the DeepRV models needed in the [heartwise-ai/DeepRV](https://huggingface.co/collections/heartwise/deeprv-673b872dcdc852f69bee89f1) repository
   - Create an API key in the HuggingFace website in `User Settings` -> `API Keys` -> `Create API Key` -> `Read`
   - Add your API key in the following format in the `api_key.json` file in the root directory:
     ```json
     {
       "huggingface_api_key": "your_api_key_here"
     }
     ```
3. 📄 Populate a csv file containing the data to be processed, example: inputs/data_rows_template.csv (see [Usage](#usage) for more details)

4. 🐳 Build the docker image:
   ```
   docker build -t deeprv-docker .
   ```

5. 🚀 Run the docker container: (see [Docker](#docker) for more details)
   ```
   docker run --gpus "device=0" --shm-size=20g -v $(pwd)/inputs:/app/inputs -v $(pwd)/outputs:/app/outputs -v $(pwd)/videos:/app/videos -i deeprv-docker --csv_file_name data_rows_template.csv
   ```

## 📄 Usage
1. Prepare your input csv file with the following columns: `FileName`, `Outcome`, `Split`, `Examen_ID` (see inputs/data_rows_template.csv for reference) 
   - Note that the `Examen_ID` column is optional for video-based metrics computation, but it is mandatory for examen-based metrics computation
   - The `FileName` column should contain the path to the DICOM files if the input is DICOM, or the path to the AVI files if the input is AVI
      - example: `/app/videos/path_to_video1.dcm` or `/app/videos/path_to_video1.avi`
2. Prepare your pipeline configuration file (see heartwise.config for reference) - **Recommended to use default config and change only the necessary parameters**
   - `model_device`: GPU device to use for inference (e.g. `cuda:0` or `cuda:1`)
   - `data_path`: Path to the input csv file **without the filename**
   - `config_path`: Path to the pipeline configuration file, **should not be modified**
   - `batch_size`: Batch size for inference
   - `output_folder`: Path to the output folder
   - `hugging_face_api_key_path`: Path to the HuggingFace API key file
   - `use_x3d`: Boolean to use X3D model
   - `use_mvit`: Boolean to use MViT model
   - `video_path`: Path to the videos folder
   - `num_workers`: Number of workers for the dataloader
   - `preprocessing_workers`: Number of workers for the preprocessing (DICOM to AVI conversion - unused if input is AVI)
   - `eval_granularity`: Granularity of the evaluation **(options: `video`, `examen`, `video examen`)**

## 🐳 Docker

### Running the Docker Container

To run the Docker container, use one of the following commands based on your hardware and mode:

**For full run:**
Run both preprocessing and analysis:
```
docker run --gpus "device=0" --shm-size=10g -v $(pwd)/inputs:/app/inputs -v $(pwd)/outputs:/app/outputs -v $(pwd)/videos:/app/videos -i deeprv-docker --csv_file_name data_rows_template.csv
```

**Without GPU (CPU only):**
```
docker run --shm-size=10g -v $(pwd)/inputs:/app/inputs -v $(pwd)/outputs:/app/outputs -v $(pwd)/videos:/app/videos -i deeprv-docker --csv_file_name data_rows_template.csv
```

These commands mount the `inputs/`, `outputs/` and `videos/` directories from your local machine to the container, allowing you to easily provide input data and retrieve results.

## 💻 Local run

1. Create a virtual environment:
   ```
   python -m venv deeprv-venv
   source deeprv-venv/bin/activate
   ```

2. Install the requirements:
   ```
   pip install -r requirements.txt
   ```

3. Install the following package manually:
   ```
   git clone https://github.com/HeartWise-AI/Orion && \
   cd Orion
   ```

4. Run the pipeline:
   Option 1: execute the main script with the correct arguments:
     ```
     python main.py --model_device cuda:0 --data_path inputs/data_rows_template.csv --config_path ./config/config_template.yaml --batch_size 32 --output_folder outputs --hugging_face_api_key_path api_key.json --use_x3d True --use_mvit False --video_path videos --num_workers 16 --eval_granularity video examen
     ```

   Option 2: execute the bash script:
     ```
     bash run_pipeline.bash --csv_file_name data_rows_template.csv
     ```

## 🤝 Contributing

Contributions to DeepECG_Docker repository are welcome! Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes and commit them with clear, descriptive messages
4. Push your changes to your fork
5. Submit a pull request to the main repository

## 📚 Citation

If you find this repository useful, please cite our work:

```
@article{,
  title={},
  author={},
  journal={},
  year={},
  publisher={}
}
```
