MODEL_MAPPING = {
    'x3d': {
        'hugging_face_model_name': 'deeprv_x3d', 
        'config' : {
            'frames': 72, 
            'resize': 256,
            'num_classes': 1,
            'model_name': 'x3d_m'
        }
    },
    'mvit': {
        'hugging_face_model_name': 'deeprv_mvit', 
        'config' : {
            'frames': 16, 
            'resize': 224, 
            'num_classes': 1,
            'model_name': 'mvit_v2_s'
        }
    },
    'swin3d_s_angio_video_classifier': {
        'hugging_face_model_name': 'swin3d_s_angio_video_classifier',
        'config': {
            'frames': 32, 
            'resize': 224, 
            'num_classes': 11,
            'model_name': 'swin3d_s'
        }
    },
    'swin3d_s_coronary_dominance_classifier': {
        'hugging_face_model_name': 'swin3d_s_coronary_dominance',
        'config': {
            'frames': 16, 
            'resize': 224, 
            'num_classes': 1,
            'model_name': 'swin3d_s'
        }
    },
    # Add more models here
}

from enum import Enum

# Angio classes
class AngioClasses(Enum):
    CATHETER = 0
    DIST_LAD = 1  # Distal Left Anterior Descending
    DIST_LCX = 2  # Distal Left Circumflex
    DIST_RCA = 3  # Distal Right Coronary Artery
    GUIDEWIRE = 4
    LAD = 5        # Left Anterior Descending
    LCX = 6        # Left Circumflex
    LEFTMAIN = 7   # Left Main Coronary Artery
    MID_LAD = 8    # Mid Left Anterior Descending
    MID_RCA = 9    # Mid Right Coronary Artery
    OBSTRUCTION = 10
    PACEMAKER = 11
    PDA = 12       # Posterior Descending Artery
    POSTEROLATERAL = 13
    PROX_RCA = 14  # Proximal Right Coronary Artery
    STENOSIS = 15
    STENT = 16
    STERNOTOMY = 17
    VALVE = 18
    
    
DICOM_TAGS = {
    'frame_height': (0x028, 0x0011),
    'frame_width': (0x028, 0x0010),
    'frame_rate': (0x08, 0x2144)
}