import torch
import numpy as np

from douzero.env.env import get_obs


def _load_model(position, model_path, model_type="lstm"):
    """
    Load the deep model for the given position.
    If model_type is 'transformer', load the transformer-based multi-modal fusion model;
    otherwise, load the original LSTM-based model.
    """
    if model_type == "transformer":
        from douzero.dmc.models import LandlordTransformerModel, FarmerTransformerModel
        if position == 'landlord':
            model = LandlordTransformerModel()
        else:
            model = FarmerTransformerModel()
    else:
        from douzero.dmc.models import LandlordLstmModel, FarmerLstmModel
        if position == 'landlord':
            model = LandlordLstmModel()
        else:
            model = FarmerLstmModel()

    model_state_dict = model.state_dict()
    if torch.cuda.is_available():
        pretrained = torch.load(model_path, map_location='cuda:0')
    else:
        pretrained = torch.load(model_path, map_location='cpu')
    pretrained = {k: v for k, v in pretrained.items() if k in model_state_dict}
    model_state_dict.update(pretrained)
    model.load_state_dict(model_state_dict)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


class DeepAgent:
    def __init__(self, position, model_path, model_type="lstm"):
        """
        Initialize the DeepAgent.

        Parameters:
            position: The agent's position ('landlord', 'landlord_up', or 'landlord_down').
            model_path: Path to the pretrained model checkpoint.
            model_type: 'lstm' or 'transformer'. Determines which model architecture to use.
        """
        self.model_type = model_type
        self.model = _load_model(position, model_path, model_type=model_type)

    def act(self, infoset):
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]

        obs = get_obs(infoset)
        z_batch = torch.from_numpy(obs['z_batch']).float()
        x_batch = torch.from_numpy(obs['x_batch']).float()
        if torch.cuda.is_available():
            z_batch, x_batch = z_batch.cuda(), x_batch.cuda()
        y_pred = self.model.forward(z_batch, x_batch, return_value=True)['values']
        y_pred = y_pred.detach().cpu().numpy()
        best_action_index = np.argmax(y_pred, axis=0)[0]
        best_action = infoset.legal_actions[best_action_index]
        return best_action
