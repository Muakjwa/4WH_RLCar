import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        # Actor ???
        self.actor_fc1 = nn.Linear(state_size, 128)
        self.actor_fc2 = nn.Linear(128, 128)
        self.actor_out = nn.Linear(128, action_size)
        
        # Critic ???
        self.critic_fc1 = nn.Linear(state_size, 128)
        self.critic_fc2 = nn.Linear(128, 128)
        self.critic_out = nn.Linear(128, 1)
        
    def forward(self, state):
        # Actor ??
        x = F.relu(self.actor_fc1(state))
        x = F.relu(self.actor_fc2(x))
        action_probs = F.softmax(self.actor_out(x), dim=-1)
        
        # Critic ??
        v = F.relu(self.critic_fc1(state))
        v = F.relu(self.critic_fc2(v))
        state_value = self.critic_out(v)
        
        return action_probs, state_value

def select_action_evaluation(state, model, device):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        action_probs, _ = model(state)
    action = torch.argmax(action_probs, dim=-1).item()
    return action

def load_model(model_path, state_size, action_size, device):
    model = ActorCritic(state_size, action_size).to(device)
    if not os.path.exists(model_path):
        print(f"Load Model from : {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only= False))
    model.eval()
    
    return model
