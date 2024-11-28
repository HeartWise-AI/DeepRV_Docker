MODEL_MAPPING = {
    'x3d': {
        'hugging_face_model_name': 'deeprv_x3d', 
        'config' : {
            'frames': 72, 
            'resize': 256, 
            'model_name': 'x3d_m'
        }
    },
    'mvit': {
        'hugging_face_model_name': 'deeprv_mvit', 
        'config' : {
            'frames': 16, 
            'resize': 224, 
            'model_name': 'mvit_v2_s'
        }
    },
    # Add more models here
}
