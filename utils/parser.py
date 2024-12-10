import argparse



class HearWiseArgs:
    @staticmethod
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Script to Run DeepRV pipeline.')
        parser.add_argument('--model_device', help='Device to run the model on', type=str, required=True)
        parser.add_argument('--data_path', help='Path to the data rows csv file and config.yaml file', type=str, required=True)
        parser.add_argument('--batch_size', help='Batch size', type=int, required=True)
        parser.add_argument('--config_path', help='Path to the config.yaml file', type=str, required=True)
        parser.add_argument('--output_folder', help='Path to the output folder', type=str, required=True)
        parser.add_argument('--hugging_face_api_key_path', help='Path to the Hugging Face API key', type=str, required=True)
        parser.add_argument('--use_x3d', help='Use X3d for video processing', type=HearWiseArgs.str2bool, required=True)
        parser.add_argument('--use_mvit', help='Use MViT for video processing', type=HearWiseArgs.str2bool, required=True)
        parser.add_argument('--video_path', help='Path to the videos', type=str, required=True)
        parser.add_argument('--num_workers', help='Number of workers for the preprocessing', type=int, default=16)
        parser.add_argument('--preprocessing_workers', help='Number of workers for the preprocessing', type=int, default=16)
        parser.add_argument('--eval_granularity', help='Granularity of the evaluation', nargs='+', default=['examen'])
        return parser.parse_args()