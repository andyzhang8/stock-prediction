import torch

model_state_dict = torch.load("final_lstm_stock_prediction_model.pth", map_location=torch.device('cpu'))
for param_tensor in model_state_dict:
    print(param_tensor, "\t", model_state_dict[param_tensor].size())
