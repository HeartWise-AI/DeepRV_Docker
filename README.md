# DeepRV_Docker


## 🚀 Features

- X3D-based multilabel classification model for video (RV function binary classification)
- MViT-based multilabel classification model for video (RV function binary classification)
- Dockerized deployment for easy setup and execution
- Configurable pipeline for flexible usage
- CPU & GPU support for accelerated processing



## 🛠️ Installation 

1. 📥 Clone the repository:
   ```
   git clone https://github.com/HeartWise-AI/DeepECG_Docker.git
   cd DeepECG_Docker
   ```

2. 🔑 Set up your HuggingFace API key:
   - Create a HuggingFace account if you don't have one yet
   - Ask for access to the DeepECG models needed in the [heartwise-ai/DeepECG](https://huggingface.co/collections/heartwise/deepecg-models-66ce09c7d620749ad819fa0d) repository
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
   docker run --gpus "device=0" -v local_path_to_inputs:/app/inputs -v local_path_to_outputs:/app/outputs -v local_path_to_videos:/app/videos -i deeprv-docker --csv_file_name data_rows_template.csv
   ```

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
     python main.py --model_device cuda:1 --data_path inputs/data_rows_template.csv --config_path ./config/config_template.yaml --batch_size 32 --output_folder outputs --hugging_face_api_key_path api_key.json --use_x3d True --use_mvit False --video_path videos --preprocessing_n_workers 16
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
