mpii_annot_path = f'../../../../Datasets/MPII/annotation/mpii_human_pose_v1_u12_1.mat'
mpii_annot_json_path = f'../../../../Datasets/MPII/annotation/mpii_annotation.json'

IMG_PATH = f'../../../../Datasets/MPII/images/'

MPII_PATH = f'../../../../Datasets/MPII/'
JSON_PATH = f'{MPII_PATH}/annotation/mpii_annotation.json'
IMG_PATH = f'{MPII_PATH}/images/'

TRAIN_CSV = f'{MPII_PATH}/annotation/train.csv'
TEST_CSV = f'{MPII_PATH}/annotation/test.csv'

columns = ['image_path',
          'x_rankle', 'y_rankle',
          'x_rknee', 'y_rknee',
          'x_rhip', 'y_rhip',
          'x_lhip', 'y_lhip',
          'x_lknee', 'y_lknee',
          'x_lankle', 'y_lankle',
          'x_pelvis', 'y_pelvis',
          'x_thorax', 'y_thorax',
          'x_upperneck', 'y_upperneck',
          'x_headtop', 'y_headtop',
          'x_rwrist', 'y_rwrist',
          'x_relbow', 'y_relbow',
          'x_rshoulder', 'y_rshoulder',
          'x_lshoulder', 'y_lshoulder',
          'x_lelbow', 'y_lelbow',
          'x_lwrist', 'y_lwrist',
          'x_human', 'y_human',
          'w_human', 'h_human']