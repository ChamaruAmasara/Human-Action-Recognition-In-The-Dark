import pandas as pd
import torch
from config.constants import (INF_CSV, INF_IMG_DIR, EXTENSION, IMG_DIM, TEST_BATCH_SIZE, NUM_CLASSES,
                                MODEL_TYPE, MODEL_NAME, DROPOUT, RNN_HIDDEN_SIZE, RNN_NUM_LAYERS, MODEL_STATE_DIR, SUBMISSION_DIR, 
                                VAL_CSV, VAL_IMG_DIR, TOP_K, INF_VID_DIR, INF_VID_FOLDER, INF_IMG_FOLDER)
from model.data import get_val_transforms, collate_fn, VideoDataset
from model.model import HARModel
from model.inference_gndtruth import inference_loop
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18, r2plus1d_18, R3D_18_Weights, R2Plus1D_18_Weights
from sklearn.metrics import accuracy_score, top_k_accuracy_score, confusion_matrix
from data_prep import sv_frame
import torch.nn as nn

sv_frame(INF_VID_DIR, INF_VID_FOLDER, INF_IMG_FOLDER)

df_test = pd.read_csv(INF_CSV, sep="\t", header=None, index_col=0)
df_test.columns = ['label', 'path']
df_test['path'] = str(INF_IMG_DIR) + '/' + df_test['path'].str.replace(EXTENSION, "")
print(df_test.head())

#df_test = pd.read_csv(INF_CSV, sep="\t", header=None, index_col=0)
#df_test.columns = ['path']
#df_test['path'] = str(INF_IMG_DIR) + '/' + df_test['path'].str.replace(EXTENSION, "")

inf_transforms = get_val_transforms(IMG_DIM)

inf_data = VideoDataset(df=df_test,
                          transforms=inf_transforms, 
                          labelAvailable=True)

inf_loader  = DataLoader(inf_data, 
                         batch_size=TEST_BATCH_SIZE,
                         shuffle=False, 
                         collate_fn=collate_fn)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if MODEL_TYPE == 'cnn-rnn':
    model = HARModel(
        model_name=MODEL_NAME,
        dropout=DROPOUT,
        rnn_hidden_size=RNN_HIDDEN_SIZE,
        rnn_num_layers=RNN_NUM_LAYERS,
        num_classes=NUM_CLASSES,
        pretrained=True)
elif MODEL_TYPE == 'r3d_18':
    model = r3d_18(weights=None, progress=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)
elif MODEL_TYPE == 'r2plus1d_18':
    model = r2plus1d_18(weights=None, progress=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)

model_paths = sorted(list(MODEL_STATE_DIR.glob('be8ab2*')))

overall_results = dict()

for m in range(len(model_paths)):

    model_path = model_paths[m]

    overall_results[m] = inference_loop(model, model_path, inf_loader, device)


for i, model_result in overall_results.items():
    model_folder, model_logits, model_pred, model_target, model_loss = model_result 
    model_score = accuracy_score(model_target, model_pred)
    model_con_mat = confusion_matrix(model_target, model_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    model_top_k_score = top_k_accuracy_score(model_target, model_logits, k=TOP_K, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(f'Model accuracy score: {model_score}')
    print(f'Model Top-{TOP_K} score: {model_top_k_score}')
    print(f'Model loss: {model_loss}')
    print(f'Confusion matrix: \n{model_con_mat}')           
    predictions = pd.Series(model_pred, name='prediction')
    predictions.to_csv(SUBMISSION_DIR / 'vr.txt', sep='\t', header=False)

   