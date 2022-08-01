import torch
import torch.nn as nn
import math
import torchvision
import torchvision.transforms as transforms
from advertorch_examples.models import LeNet5Madry
from utils.fab import FABAttack_PT
from utils.square import SquareAttack
import argparse


from collections import OrderedDict
def fix_key(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v
    return new_state_dict


def loader(data_dir, bs=250):
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = torchvision.datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(mnist, batch_size=bs, shuffle=False, num_workers=2)
    return loader


def DLRLoss(logits, label, device="cuda"):
    length = logits.shape[0]
    z_y = logits[range(length), label] # 正解ラベルのlogit
    z_max = torch.zeros(length).to(device) # 正解ラベルを除く中で最大のlogit
    z_pi1 = logits.data.max(1)[0]
    z_pi3 = torch.topk(logits, 3, dim=1)[0][:,2]
  
    for i in range(length):
        if logits[i].data.max(0)[1] == label[i]: # maxが正解ラベルだったとき
            z_max[i] = torch.topk(logits[i], 2)[0][1] # 2番目に大きいlogits
        else:
            z_max[i] = torch.max(logits[i])
    
    loss = -(z_y - z_max)/(z_pi1 - z_pi3)
    
    return torch.sum(loss)/loss.shape[0]


def TargetedDLRLoss(logits, label, targets): # 分母に+eps
    eps = 1e-12
    length = logits.shape[0]
    
    z_y = logits[range(length), label] # 正解ラベルのlogit
    z_t = logits[range(length), targets]
    topk_z = torch.topk(logits, 4, dim=1)[0]
    z_pi1 = topk_z[:,0]
    z_pi3 = topk_z[:,2]
    z_pi4 = topk_z[:,3]
    
    loss = -(z_y - z_t)/(z_pi1 - (z_pi3 + z_pi4)/2 + eps)
    
    return torch.sum(loss)/loss.shape[0]


class APGD_CE():
    def __init__(self, model, eps=0.03, rho=0.75, N_iter=100, alpha=0.75, device="cuda"):
        self.model = model
        self.eps = eps
        self.rho = rho
        self.N_iter = N_iter
        self.alpha = alpha
        self.loss = nn.CrossEntropyLoss()
        self.device = device
        self.w = []
        # iterationごとに目的関数の値のリスト
        self.f_list = []
        # iterationごとのetaのリスト
        self.etas = []
        # iterationごとに目的関数の最大値のリスト
        self.f_maxlist = []
        self.p = [0, 0.22]
        self.checkpoints()
        
    def condition1(self, j):
        count = 0
        for i in range(self.w[j-1],self.w[j]):
            if self.f_list[i+1] > self.f_list[i]:
                count  += 1
        if count < self.rho*(self.w[j] - self.w[j-1]):
            return True
        return False
    
    def condition2(self, j):
        if self.etas[self.w[j-1]] == self.etas[self.w[j]] and self.f_maxlist[self.w[j-1]] == self.f_maxlist[self.w[j]]:
            return True
        return False
    
    def checkpoints(self):
        w = []
        while True:
            next_p = self.p[-1] + max(self.p[-1] - self.p[-2] - 0.03, 0.06)
            if next_p > 1:
                break
            self.p.append(next_p)
        for pj in self.p:
            w.append(math.ceil(pj * self.N_iter))
        self.w = w
    
    def perturb(self, x, y):
        x_orig = x.clone().detach()
        x_0 = x.clone().detach()
        x_0.requires_grad = True

        # 目的関数fの初期値
        self.model.zero_grad()
        logit_0 = self.model(x_0)
        f_0 = self.loss(logit_0, y)
        f_0.backward()

        # 目的関数fの次の値
        eta = 2*self.eps
        x_1pre = x_0 + eta*x_0.grad.sign()
        noize_x1 = torch.clamp(x_1pre - x_orig, min=-self.eps, max=self.eps)
        x_1 = torch.clamp(x_orig + noize_x1, min=0.0, max=1.0).clone().detach()
        x_1.requires_grad = True
        self.model.zero_grad()
        logit_1 = self.model(x_1)
        f_1 = self.loss(logit_1, y)
        f_1.backward()

        f_max = max(f_0, f_1)
        x_max = (x_orig if f_max == f_0 else x_1)
        eta_0 = 2*self.eps

        self.f_list = [f_0, f_1]
        self.f_maxlist = [f_0, f_max]
        self.etas = [eta_0, eta_0]

        f_xk = f_1
        x_k = x_1
        x_km = x_orig

        wj = 1
        for k in range(1, self.N_iter):
            # calculate z
            z_pre = x_k + self.etas[-1]*x_k.grad.sign()
            noise_z = torch.clamp(z_pre - x_orig, min=-self.eps, max=self.eps)
            z = torch.clamp(x_orig + noise_z, min=0.0, max=1.0)

            # calculate x_kp := x^(k+1)
            x_pre = x_k + self.alpha*(z - x_k) + (1 - self.alpha)*(x_k - x_km)
            noize_x = torch.clamp(x_pre - x_orig, min=-self.eps, max=self.eps)
            x_kp = torch.clamp(x_orig + noize_x, min=0.0, max=1.0).clone().detach()
            x_kp.requires_grad = True

            # update x,f max
            logit = self.model(x_kp)
            f_xkp = self.loss(logit, y)
            f_xkp.backward()
            if f_xkp > f_max:
                x_max = x_kp
                f_max = f_xkp

            # update eta
            if k in self.w:
                if self.condition1(wj) or self.condition2(wj):
                    eta = eta/2
                    x_kp = x_max
                wj += 1

            # update x, x_km
            x_km = x_k
            x_k = x_kp

            # append eta, f, f_max
            self.etas.append(eta)
            self.f_list.append(f_xkp)
            self.f_maxlist.append(f_max)

        return x_max
    

class APGD_Targeted():
    def __init__(self, model, eps=0.03, rho=0.75, N_iter=100, alpha=0.75, loss="DLR", device="cuda"):
        self.model = model
        self.eps = eps
        self.rho = rho
        self.N_iter = N_iter
        self.alpha = alpha
        self.loss = (nn.CrossEntropyLoss() if loss == "CE" else TargetedDLRLoss)
        self.device = device
        self.w = []
        # iterationごとに目的関数の値のリスト
        self.f_list = []
        # iterationごとのetaのリスト
        self.etas = []
        # iterationごとに目的関数の最大値のリスト
        self.f_maxlist = []
        self.p = [0, 0.22]
        # targetの対象にするクラスの数
        self.num_target_class = 9
        self.checkpoints()
        
    def condition1(self, j):
        count = 0
        for i in range(self.w[j-1],self.w[j]):
            if self.f_list[i+1] > self.f_list[i]:
                count  += 1
        if count < self.rho*(self.w[j] - self.w[j-1]):
            return True
        return False
    
    def condition2(self, j):
        if self.etas[self.w[j-1]] == self.etas[self.w[j]] and self.f_maxlist[self.w[j-1]] == self.f_maxlist[self.w[j]]:
            return True
        return False
    
    def checkpoints(self):
        w = []
        while True:
            next_p = self.p[-1] + max(self.p[-1] - self.p[-2] - 0.03, 0.06)
            if next_p > 1:
                break
            self.p.append(next_p)
        for pj in self.p:
            w.append(math.ceil(pj * self.N_iter))
        self.w = w
    
    def one_attack(self, x, y, target):
        x_orig = x.clone().detach()
        x_0 = x.clone().detach()
        x_0.requires_grad = True

        # 目的関数fの初期値
        self.model.zero_grad()
        logit_0 = self.model(x_0)
        f_0 = self.loss(logit_0, y, target)
        f_0.backward()

        # 目的関数fの次の値
        eta = 2*self.eps
        x_1pre = x_0 + eta*x_0.grad.sign()
        noize_x1 = torch.clamp(x_1pre - x_orig, min=-self.eps, max=self.eps)
        x_1 = torch.clamp(x_orig + noize_x1, min=0.0, max=1.0).clone().detach()
        x_1.requires_grad = True
        self.model.zero_grad()
        logit_1 = self.model(x_1)
        f_1 = self.loss(logit_1, y, target)
        f_1.backward()


        f_max = max(f_0, f_1)
        x_max = (x_orig if f_max == f_0 else x_1)
        eta_0 = 2*self.eps

        self.f_list = [f_0, f_1]
        self.f_maxlist = [f_0, f_max]
        self.etas = [eta_0, eta_0]

        f_xk = f_1
        x_k = x_1
        x_km = x_orig

        wj = 1
        for k in range(1, self.N_iter):
            # calculate z
            z_pre = x_k + self.etas[-1]*x_k.grad.sign()
            noise_z = torch.clamp(z_pre - x_orig, min=-self.eps, max=self.eps)
            z = torch.clamp(x_orig + noise_z, min=0.0, max=1.0)

            # calculate x_kp := x^(k+1)
            x_pre = x_k + self.alpha*(z - x_k) + (1 - self.alpha)*(x_k - x_km)
            noize_x = torch.clamp(x_pre - x_orig, min=-self.eps, max=self.eps)
            x_kp = torch.clamp(x_orig + noize_x, min=0.0, max=1.0).clone().detach()
            x_kp.requires_grad = True

            # update x,f max
            logit = self.model(x_kp)
            f_xkp = self.loss(logit, y, target)
            f_xkp.backward()
            if f_xkp > f_max:
                x_max = x_kp
                f_max = f_xkp

            # update eta
            if k in self.w:
                if self.condition1(wj) or self.condition2(wj):
                    eta = eta/2
                    x_kp = x_max
                wj += 1

            # update x, x_km
            x_km = x_k
            x_k = x_kp

            # append eta, f, f_max
            self.etas.append(eta)
            self.f_list.append(f_xkp)
            self.f_maxlist.append(f_max)

        return x_max

    def perturb(self, x, y):
        x_adv = x.clone().detach()
        x_adv_tmp = x.clone().detach()
        outputs = self.model(x_adv_tmp)
        correct_flag = torch.tensor([True]*x.shape[0]).to(self.device)
        # まだ正解している（攻撃対象の）画像のインデックス
        target_ind = torch.eq(outputs.data.max(1)[1], y).nonzero().squeeze()
        # モデルが正解している画像のインデックスがTrueとなるフラグのリスト
        correct_flag = torch.eq(correct_flag, torch.eq(outputs.data.max(1)[1], y))
        record_target_class = outputs.sort()[1]
        for target in range(2, self.num_target_class+2):
            if len(target_ind) == 0:
                # 攻撃対象の画像がなければ終了
                break
            x_tobe_attacked = (x[target_ind]).clone().detach()
            y_remain = y[target_ind]
            target_class = record_target_class[target_ind, -target]
            x_adv_tmp = self.one_attack(x_tobe_attacked, y_remain, target_class)

            outputs = self.model(x_adv_tmp)
            # 攻撃に成功した画像のインデックス
            success_ind = (~torch.eq(outputs.data.max(1)[1], y_remain)).nonzero().squeeze()

            x_adv[target_ind[success_ind]] = x_adv_tmp[success_ind]
            correct_flag[target_ind[success_ind]] = False
            target_ind = correct_flag.nonzero().squeeze()

        return x_adv


def autoattack(model, testloader, eps=8/255, bs=250, device="cuda"):
    attacks = {
        "APGD": APGD_CE(model, eps=eps, device=device),
        "APGD_T": APGD_Targeted(model, eps=eps, device=device),
        "FAB": FABAttack_PT(model, n_restarts=5, n_iter=100, eps=eps, seed=0, norm="Linf", verbose=False, device=device),
        "Square": SquareAttack(model, p_init=0.8, n_queries=5000, eps=eps, norm="Linf", n_restarts=1, seed=0, verbose=False, device=device, resc_schedule=False)
    }
    attacks["FAB"].targeted = True
    attacks["FAB"].n_restarts = 1
    attacks["FAB"].seed = 0
    attacks["Square"].seed = 0

    correct, whole = [0]*4, 10000
    ans_flag = torch.tensor([True]*10000).to(device)
    for i, (data, label) in enumerate(testloader):
        for j, (name, attack) in enumerate(attacks.items()):
            if torch.sum(ans_flag[i*bs:(i+1)*bs]) == 0:
                break
            x = data.to(device)
            y = label.to(device)
            adv_images = attack.perturb(x, y)
            output = model(adv_images)

            correct_flag = torch.eq(output.data.max(1)[1], y)
            correct[j] += torch.sum(correct_flag)

            ans_flag[i*bs:(i+1)*bs] = torch.logical_and(ans_flag[i*bs:(i+1)*bs], correct_flag)

    print("Robust Accuracy:", ans_flag.sum()/len(ans_flag))


def mma_mnist_eval(model_path, data_path, device):
    # MMA Trainingのモデルの評価
    # https://github.com/BorealisAI/mma_training
    state_dict = torch.load(model_path)
    model = LeNet5Madry()
    model.load_state_dict(state_dict["model"])
    model.to(device)
    model.eval()
    
    testloader = loader(data_path, bs=250)
    whole, correct = 0, 0
    with torch.no_grad():
        for data, label in testloader:
            data, label = data.to(device), label.to(device)
            logits = model(data)
            correct += (logits.data.max(1)[1] == label).sum()
            whole += data.shape[0]
    print("Clean Accuracy:", correct/whole)
    autoattack(model, testloader, eps=0.3, bs=250, device=device)


def set_seed():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def main(args):
    set_seed()
    mma_mnist_eval(model_path=args.model, data_path=args.data_dir, device=args.device)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    main(args)