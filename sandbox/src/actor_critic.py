import numpy as np
import torch
from tqdm import tqdm
import yaml
from pathlib import Path

IMG_SIZE = 32
AGENTS_NUM = 5
SCEN_NUM = 1000
HORIZONT = 5
GAMMA = 0.1
ENTROPY_COEFF = 0.5

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def conv_block(in_channels, out_channels, kernel_size, batch_norm=None,
               drop_out_rate=None, max_pool=None):
    block_list = [torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1)]

    if batch_norm:
        block_list.append(torch.nn.BatchNorm2d(out_channels))

    block_list.append(torch.nn.LeakyReLU())

    if drop_out_rate:
        block_list.append(torch.nn.Dropout2d(p=drop_out_rate))

    if max_pool:
        block_list.append(torch.nn.MaxPool2d(max_pool, stride=max_pool))

    return block_list


def linear_block(input_dim, output_dim, batch_norm=None, drop_out_rate=None):
    block_list = [torch.nn.Linear(input_dim, output_dim, bias=True)]

    if batch_norm:
        block_list.append(torch.nn.BatchNorm1d(num_features=output_dim))

    block_list.append(torch.nn.LeakyReLU())

    if drop_out_rate:
        block_list.append(torch.nn.Dropout(p=drop_out_rate))

    return block_list


def add_curve_to_image(img, points, cx=1):
    for t in np.arange(0, 1, 0.01):
        # x_curve = ((1 - t) ** 2 * points[0][0] + 2 * t * (1 - t) * points[0][1] + t ** 2 * points[0][2])
        x_curve = (1 - t)*points[0] + t*points[1]
        x_curve = int(np.floor(x_curve))
        # y_curve = ((1 - t) ** 2 * points[0][3] + 2 * t * (1 - t) * points[0][4] + t ** 2 * points[0][5])
        y_curve = (1 - t)*points[2] + t*points[3]
        y_curve = int(np.floor(y_curve))
        img[y_curve, x_curve] = cx

    return img


def calc_loss(img, aim_image, original_code=True):
    if original_code:
        loss = np.mean(np.power(aim_image - img, 2))
    else:
        raise NotImplemented
        # @TODO pixel-wise loss calculation: ONLY for naive approach with Bezier curves

    return loss


def calc_reward(loss, prev_loss, original_code=True):
    if original_code:
        if loss < prev_loss:  # original code
            reward = (prev_loss - loss) / prev_loss
        else:
            reward = (prev_loss - loss) / loss
    else:
        reward = np.exp(np.abs(prev_loss - loss))
        if loss > prev_loss:  # set penalty - reverse reward
            reward = -reward
    return reward


class Model(torch.nn.Module):
    def __init__(self, output_size, conv_layers, linear_layers=None, adaptive_pooling_size=16):
        super().__init__()

        layers_list = []
        for key, val in conv_layers.items():
            layers_list += conv_block(**val)

        layers_list = torch.nn.Sequential(*layers_list)
        self.conv_blocks = torch.nn.ModuleList(layers_list)
        self.conv_output_channels = val['out_channels']
        self.adapt = torch.nn.AdaptiveMaxPool2d(adaptive_pooling_size)

        self.linear_blocks = None
        if linear_layers:
            layers_list = []
            for n, (key, val) in enumerate(linear_layers.items()):
                if n == 0:
                    layers_list += linear_block(
                        self.conv_output_channels * adaptive_pooling_size ** 2,
                        val['output_dim'], val['batch_norm'], val['drop_out_rate']
                    )
                else:
                    layers_list += linear_block(**val)
            conv_output_size = val['output_dim']

            layers_list = torch.nn.Sequential(*layers_list)
            self.linear_blocks = torch.nn.ModuleList(layers_list)
        else:
            conv_output_size = adaptive_pooling_size ** 2

        self.output_point_one = torch.nn.Linear(conv_output_size, output_size)
        self.output_point_two = torch.nn.Linear(conv_output_size + output_size, output_size)
        self.value = torch.nn.Linear(conv_output_size + 2 * output_size, 1)
        self.output_activation = torch.nn.Softmax(dim=1)

    def forward(self, x):

        for block in self.conv_blocks:
            x = block(x)

        x = self.adapt(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])

        if self.linear_blocks:
            for block in self.linear_blocks:
                x = block(x)

        point_one_policy = self.output_point_one(x)

        x = torch.cat((x, point_one_policy), dim=1)
        point_two_policy = self.output_point_two(x)

        x = torch.cat((x, point_two_policy), dim=1)
        value = self.value(x)

        point_one_policy = self.output_activation(point_one_policy)
        point_two_policy = self.output_activation(point_two_policy)

        return {
            'point_one_policy': point_one_policy,
            'point_two_policy': point_two_policy,
            'value': value
        }


def load_configs(config_path, split):
    with open(config_path, 'r') as stream:
        configs = yaml.safe_load(stream)[split]
        conv_layers = configs['conv_layers']
        linear_layers = configs['linear_layers']
    return conv_layers, linear_layers


def test(config_path='default_configs.yaml'):
    conv_layers, linear_layers = load_configs(config_path, split='test')
    model = Model(IMG_SIZE * IMG_SIZE, conv_layers, linear_layers)
    model.to(device)
    img_torch = torch.rand(1, 2, IMG_SIZE, IMG_SIZE).float().to(device)
    x = model.forward(img_torch)
    return x


def train(config_path='default_configs.yaml'):
    conv_layers, linear_layers = load_configs(config_path, split='train')
    model = Model(IMG_SIZE * IMG_SIZE, conv_layers, linear_layers)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    history_loss = []
    history_agent_reward = []
    history_agent_loss = []
    actor_component_loss = []
    value_component_loss = []
    entropy_component_loss = []
    for scen in tqdm(range(SCEN_NUM)):

        scen_flag = True
        aim_images_list = []
        pred_images_list = []
        prev_loss_list = []
        for x in range(AGENTS_NUM):
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
            img = img.astype(np.float32)

            start_line = int(np.random.rand() * 2)
            for i in range(0, IMG_SIZE, int(IMG_SIZE / 10)):
                img[(start_line + i): (start_line + i + 1), :] = 1

            aim_images_list.append(np.copy(img))
            aim_img = np.copy(img)

            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
            img = img.astype(np.float32)
            pred_images_list.append(img)

            prev_loss_list.append(calc_loss(img, aim_img))

        agent_status_list = [True for x in range(AGENTS_NUM)]

        scen_res_dict = dict()

        t = 0
        while scen_flag:
            scen_res_dict[t] = list()
            for x in range(AGENTS_NUM):
                scen_res_dict[t].append(dict())

            for ag in range(AGENTS_NUM):

                if agent_status_list[ag]:
                    img_torch = torch.zeros(1, 2, IMG_SIZE, IMG_SIZE).float().to(device)
                    img_torch[0, 0, :, :] = torch.tensor(aim_images_list[ag])
                    img_torch[0, 1, :, :] = torch.tensor(np.copy(pred_images_list[ag]))
                    img_torch.to(device)

                    model_output = model.forward(img_torch)

                    point_one_policy = torch.max(model_output['point_one_policy'])
                    point_two_policy = torch.max(model_output['point_two_policy'])

                    policy = point_one_policy * point_two_policy
                    log_policy = torch.log(policy)

                    point_one_index = int(torch.argmax(model_output['point_one_policy']))
                    point_two_index = int(torch.argmax(model_output['point_two_policy']))

                    point_one_Y_coord = point_one_index // IMG_SIZE
                    point_one_X_coord = point_one_index - point_one_Y_coord * IMG_SIZE

                    point_two_Y_coord = point_two_index // IMG_SIZE
                    point_two_X_coord = point_two_index - point_two_Y_coord * IMG_SIZE

                    entropy = torch.matmul(
                        torch.transpose(model_output['point_one_policy'], 0, 1),
                        model_output['point_two_policy']
                    )
                    entropy = torch.sum(entropy * torch.log(entropy))

                    points = [point_one_X_coord, point_two_X_coord, point_one_Y_coord, point_two_X_coord]

                    pred_img = np.copy(pred_images_list[ag])
                    pred_img = add_curve_to_image(pred_img, points)

                    loss = calc_loss(pred_img, aim_images_list[ag])
                    reward = calc_reward(loss, prev_loss_list[ag], points)

                    scen_res_dict[t][ag]['log_policy'] = log_policy
                    scen_res_dict[t][ag]['value'] = model_output['value']
                    scen_res_dict[t][ag]['reward'] = reward
                    scen_res_dict[t][ag]['entropy'] = entropy
                    scen_res_dict[t][ag]['active_agent_flag'] = True
                    scen_res_dict[t][ag]['image_loss'] = loss

                    pred_images_list[ag] = pred_img
                    prev_loss_list[ag] = loss

            if t > HORIZONT:
                loss_list = list()
                for ag in range(AGENTS_NUM):
                    if agent_status_list[ag]:

                        t0 = t - HORIZONT + 1
                        G_t = 0
                        for k in range(t0, t):
                            reward = scen_res_dict[k][ag]['reward']
                            G_t = G_t + GAMMA ** (k - t0) * reward

                        G_t = G_t + GAMMA ** (t - t0) * scen_res_dict[t][ag]['value']
                        Advantage = G_t - scen_res_dict[t0][ag]['value']

                        actor_loss = Advantage * scen_res_dict[t0][ag]['log_policy']
                        critic_loss = Advantage * scen_res_dict[t0][ag]['value']
                        entropy = scen_res_dict[t0][ag]['entropy']

                        loss = -(actor_loss + critic_loss + ENTROPY_COEFF * entropy)
                        loss = -(actor_loss + ENTROPY_COEFF * entropy)
                        loss_list.append(loss)

                        print(t, ag)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        if ag == 0:
                            actor_component_loss.append(float(actor_loss.cpu().detach().numpy()))
                            value_component_loss.append(float(critic_loss.cpu().detach().numpy()))
                            entropy_component_loss.append(float(entropy.cpu().detach().numpy()))

                        if scen_res_dict[t][ag]['image_loss'] < 0.01:
                            agent_status_list[ag] = False

                history_loss.append(float(loss.cpu().detach().numpy()))

            history_agent_reward.append(scen_res_dict[t][0]['reward'])
            history_agent_loss.append(scen_res_dict[t][0]['image_loss'])

            if t > 20:
                scen_flag = False

            k = 0
            for ag in range(AGENTS_NUM):
                if not agent_status_list[ag]:
                    k += 1

            if k == AGENTS_NUM:
                scen_flag = False
            t += 1


def main():
    train()
    # test()


if __name__ == '__main__':
    main()
