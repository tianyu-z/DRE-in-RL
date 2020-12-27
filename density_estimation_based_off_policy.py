# This script is developed based on https://github.com/seungeunrho/minimalRL
import gym
import random
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# Characteristics
# 1. Discrete action space, single thread version.
# 2. Does not support trust-region updates.
# Hyperparameters
learning_rate = 0.0002
gamma = 0.98
buffer_limit = 6000
rollout_len = 10
batch_size = (
    4  # Indicates 4 sequences per mini-batch (4*rollout_len = 40 samples total)
)
c = 1.0  # For truncating importance sampling ratio


def init_weights(net, init_type="normal", init_gain=0.02, debug=False):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    from torch.nn import init

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if debug:
                print(classname)
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif (
            classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


class ReplayBuffer:
    """
    The Buffer for the off-policy training
    """

    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, seq_data):
        self.buffer.append(seq_data)

    def sample(self, on_policy=False):
        if on_policy:
            mini_batch = [self.buffer[-1]]
        else:
            mini_batch = random.sample(self.buffer, batch_size)
        s_lst, a_lst, r_lst, prob_lst, done_lst, is_first_lst = [], [], [], [], [], []
        for seq in mini_batch:
            is_first = True  # Flag for indicating whether the transition is the first item from a sequence
            for transition in seq:
                s, a, r, prob, done = transition
                s_lst.append(s)
                a_lst.append([a])
                r_lst.append(r)
                prob_lst.append(prob)
                done_mask = 0.0 if done else 1.0
                done_lst.append(done_mask)
                is_first_lst.append(is_first)
                is_first = False
        s, a, r, prob, done_mask, is_first = (
            torch.tensor(s_lst, dtype=torch.float),
            torch.tensor(a_lst),
            r_lst,
            torch.tensor(prob_lst, dtype=torch.float),
            done_lst,
            is_first_lst,
        )
        return s, a, r, prob, done_mask, is_first

    def size(self):
        return len(self.buffer)


class LSTM(nn.Module):
    """
    The lstm domain classifer
    """

    def __init__(
        self, lstm_input_dim, lstm_num_layer, lstm_hidden_dim, lstm_output_dim
    ):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            batch_first=True,
            num_layers=lstm_num_layer,
        )
        self.fc = nn.Linear(lstm_hidden_dim, lstm_output_dim)
        self.lstm_input_dim = lstm_input_dim
        self.lstm_num_layer = lstm_num_layer
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_output_dim = lstm_output_dim
        self.sigmoid = torch.nn.Sigmoid()

    def init_hidden(self, batch_size):
        self.hidden = (
            torch.zeros(self.lstm_num_layer, batch_size, self.lstm_hidden_dim),
            torch.zeros(self.lstm_num_layer, batch_size, self.lstm_hidden_dim),
        )

    def forward(self, X):
        X, self.hidden = self.lstm(X)
        X = self.fc(X)
        X = self.sigmoid(X)
        return X


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_q = nn.Linear(256, 2)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        pi = F.softmax(x, dim=softmax_dim)
        return pi

    def q(self, x):
        x = F.relu(self.fc1(x))
        q = self.fc_q(x)
        return q


def train(model, optimizer, memory, on_policy=False, lstm=None):
    s, a, r, prob, done_mask, is_first = memory.sample(on_policy)
    if lstm is not None:
        # prepare the data to create the ratio
        tensor_list = []
        for i in range(len(a)):
            s0 = s[i].clone().cpu().numpy()
            s0 = np.append(s0, a[i])
            s0 = np.append(s0, r[i])
            s0 = np.append(s0, prob[i])
            tensor_list.append(torch.tensor(s0))
        tensor_input = torch.vstack(tensor_list).unsqueeze(1)
        prediction = lstm(tensor_input)
        dynamic_ratio = prediction / (1 - prediction)
        dynamic_ratio = dynamic_ratio.detach()
        dynamic_ratio = dynamic_ratio[:, :, 0].repeat(1, 2)
    else:
        dynamic_ratio = 1
    q = model.q(s)
    q_a = q.gather(1, a)
    pi = model.pi(s, softmax_dim=1)
    pi_a = pi.gather(1, a)
    v = (q * pi).sum(1).unsqueeze(1).detach()
    rho = dynamic_ratio * pi.detach() / prob
    rho_a = rho.gather(1, a)
    rho_bar = rho_a.clamp(max=c)
    correction_coeff = (1 - c / rho).clamp(min=0)
    q_ret = v[-1] * done_mask[-1]
    q_ret_lst = []
    for i in reversed(range(len(r))):
        q_ret = r[i] + gamma * q_ret
        q_ret_lst.append(q_ret.item())
        q_ret = rho_bar[i] * (q_ret - q_a[i]) + v[i]
        if is_first[i] and i != 0:
            q_ret = (
                v[i - 1] * done_mask[i - 1]
            )  # When a new sequence begins, q_ret is initialized
    q_ret_lst.reverse()
    q_ret = torch.tensor(q_ret_lst, dtype=torch.float).unsqueeze(1)
    loss1 = -rho_bar * torch.log(pi_a) * (q_ret - v)
    loss2 = (
        -correction_coeff * pi.detach() * torch.log(pi) * (q.detach() - v)
    )  # bias correction term
    loss = loss1 + loss2.sum(1) + F.smooth_l1_loss(q_a, q_ret)
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()


def main():
    env = gym.make("CartPole-v1")
    memory = ReplayBuffer()

    # initialize models incide FFNN for the ActorCritic and lstm for the domain classifier
    model = ActorCritic()
    lstm = LSTM(8, 5, 128, 1).double()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer_lstm = optim.Adam(lstm.parameters(), lr=learning_rate)
    init_weights(model, init_type="xavier")
    init_weights(lstm, init_type="xavier")
    mse_loss = torch.nn.MSELoss()

    score = 0.0
    print_interval = 1
    save_interval = 100
    max_epi = 10000

    # saving checkpoints
    save_dict = {}
    save_dict["n_epi"] = []
    save_dict["score"] = []

    for n_epi in range(max_epi):
        s = env.reset()
        done = False

        # train the lstm domain classifier after some episode because we need to collect some tensor data for training
        if n_epi > 100:
            print("Training the lstm classifier.")
            optimizer_lstm.zero_grad()
            min_len = min([len(tensor_data_old), len(tensor_data)])
            tensor_data_old_train = tensor_data_old[:min_len]
            tensor_data = tensor_data[:min_len]
            label = [[0] * min_len + [1] * min_len]
            if not isinstance(tensor_data_old_train, torch.Tensor):
                tensor_data_old_train = torch.vstack((tensor_data_old_train))
            if not isinstance(tensor_data, torch.Tensor):
                tensor_data = torch.vstack((tensor_data))
            input_data = (
                torch.vstack((tensor_data_old_train, tensor_data)).unsqueeze(0).double()
            )
            input_label = torch.tensor(label).unsqueeze(-1).double()
            prediction = lstm(input_data)
            loss = mse_loss(prediction, input_label)
            loss.mean().backward()
            optimizer_lstm.step()

        # initialize tensor data buffer
        if n_epi == 0:
            tensor_data_old = []
            tensor_data = []
        # update tensor data buffer
        else:
            tensor_data_old = tensor_data
            tensor_data = []

        while not done:
            seq_data = []
            for t in range(rollout_len):
                prob = model.pi(torch.from_numpy(s).float())
                a = Categorical(prob).sample().item()
                s_prime, r, done, info = env.step(a)
                prob_numpy = prob.detach().numpy()
                seq_data.append((s, a, r / 100.0, prob_numpy, done))

                # collect input tensor for lstm
                s0 = s.copy()
                s0 = np.append(s0, a)
                s0 = np.append(s0, r / 100)
                s0 = np.append(s0, prob_numpy)
                tensor_data.append(torch.tensor(s0))

                score += r
                s = s_prime
                if done:
                    break
            memory.put(seq_data)
            # if episode<=100, train without lstm
            if memory.size() > 500 and n_epi <= 100:
                train(model, optimizer, memory)
            # if episode>100, train with the lstm as the domain classifer
            elif memory.size() > 500 and n_epi >= 100:
                train(model, optimizer, memory, lstm=lstm)

        if n_epi % print_interval == 0 and n_epi != 0:
            print(
                "# of episode :{}, avg score : {:.1f}, buffer size : {}".format(
                    n_epi, score / print_interval, memory.size()
                )
            )
            save_dict["n_epi"].append(n_epi)
            save_dict["score"].append(score)
            score = 0.0

        if n_epi % save_interval == 0 and n_epi != 0:
            save_dict["model"] = model.cpu().state_dict()
            save_dict["lstm"] = lstm.cpu().state_dict()
            torch.save(save_dict, "ckpt_{}_with_lstm.pth".format(n_epi))
    env.close()


if __name__ == "__main__":
    main()
