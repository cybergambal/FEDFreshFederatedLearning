
import os
import time
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import copy 

from byzfl import DataDistributor

def get_data_loaders(data_mode, batch_size, num_users):
    """
    Get data loaders for training and testing datasets.
    
    :param data_mode: Type of dataset ("MNIST" or "CIFAR")
    :param batch_size: Batch size for data loaders
    :param num_users: Number of users for data distribution
    """
    if data_mode == "MNIST" or data_mode == "Regression":
        # MNIST dataset and preprocessing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        # CIFAR-10 dataset and preprocessing
        transform1 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                            std=[0.229, 0.224, 0.225])
        ])
        
        transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                            std=[0.229, 0.224, 0.225])
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform1)
        trainset.targets = torch.Tensor(trainset.targets).long()
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform2)

    num_cores = os.cpu_count()
    num_workers = min(4, num_cores) if num_cores else 0

    data_loader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=True, num_workers=num_workers)
    
    labels_group1 = [0,1,2,3,4]
    labels_group2 = [5,6,7,8,9]

    indices_group1 = [i for i, label in enumerate(trainset.targets) if label in labels_group1]
    indices_group2 = [i for i, label in enumerate(trainset.targets) if label in labels_group2]

    trainset_group1 = torch.utils.data.Subset(trainset, indices_group1)
    trainset_group2 = torch.utils.data.Subset(trainset, indices_group2)
    
    params_group1 = {
        "data_distribution_name": "iid",
        "nb_honest": num_users//2,
        "data_loader": trainset_group1,
        "batch_size": batch_size,
    }

    params_group2 = {
        "data_distribution_name": "iid",
        "nb_honest": num_users - num_users//2,
        "data_loader": trainset_group2,
        "batch_size": batch_size,
    }
    whole_params_group1 = {
        "data_distribution_name": "iid",
        "nb_honest": num_users//2,
        "data_loader": trainset_group1,
        "batch_size": len(trainset_group1) // (num_users//2),
    }

    whole_params_group2 = {
        "data_distribution_name": "iid",
        "nb_honest": num_users - num_users//2,
        "data_loader": trainset_group2,
        "batch_size": len(trainset_group2) // (num_users - num_users//2),
    }
    distributor_group1 = DataDistributor(params_group1)
    distributor_group2 = DataDistributor(params_group2)
    whole_distributor_group1 = DataDistributor(whole_params_group1)
    whole_distributor_group2 = DataDistributor(whole_params_group2)
    WholeTrainSetUsers = whole_distributor_group1.split_data()
    WholeTrainSetUsers.extend(whole_distributor_group2.split_data())
    TrainSetUsers = distributor_group1.split_data()
    TrainSetUsers.extend(distributor_group2.split_data())
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(len(trainset), "training samples loaded.")
    print(len(TrainSetUsers[0].dataset), "samples allocated to each user.")
    assert len(trainset) >= num_users * (len(trainset) // num_users), "Dataset too small for requested user allocation!"

    return TrainSetUsers, testloader, data_loader, WholeTrainSetUsers

def get_Model(data_mode, train_mode='all'):

    """
    Get the model class based on the dataset and training mode.
    
    :param data_mode: Dataset mode ("Regression", "MNIST" or "CIFAR")
    :param train_mode: Training mode ("all", "dense", or "conv")
    """

    if data_mode == "MNIST":
        # CustomCNN Model
        class Model(nn.Module):
            def __init__(self, num_classes=10, train_mode=train_mode):
                """
                train_mode: 
                    'all'    → train everything
                    'dense'  → train only fc1 and fc2
                    'conv'   → train only conv1 and conv2
                """
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.pool1 = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.pool2 = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(64 * 7 * 7, 128)
                self.fc2 = nn.Linear(128, num_classes)

                # Set requires_grad according to training mode
                if train_mode == 'dense':
                    for param in self.conv1.parameters(): param.requires_grad = False
                    for param in self.conv2.parameters(): param.requires_grad = False
                elif train_mode == 'conv':
                    for param in self.fc1.parameters(): param.requires_grad = False
                    for param in self.fc2.parameters(): param.requires_grad = False
                # 'all' means train everything → no changes needed

            def forward(self, x):
                x = self.pool1(F.relu(self.conv1(x)))
                x = self.pool2(F.relu(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                return self.fc2(x)
    
    elif data_mode == "Regression":
        class Model(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                self.fc1 = nn.Linear(28*28, num_classes)

            def forward(self, x):
                x = x.view(x.size(0), -1)
                out = F.sigmoid(self.fc1(x))
                return out
        

    elif data_mode == "CIFAR":
        class ResidualBlock(nn.Module):
            def __init__(self, inchannel, outchannel, stride=1):
                super(ResidualBlock, self).__init__() 
                self.dropout = nn.Dropout()
                self.left = nn.Sequential(
                    nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.GroupNorm(32,outchannel),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False), 
                    nn.GroupNorm(32,outchannel),
                )
                self.shortcut = nn.Sequential()
                if stride != 1 or inchannel != outchannel:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False), 
                        nn.GroupNorm(32,outchannel),
                    )
                    
            def forward(self, x):
                out = self.left(x)
                out = out + self.shortcut(x)
                #out = self.dropout(out)
                out = F.relu(out)
                
                return out

        class ResNet(nn.Module):
            def __init__(self, ResidualBlock, num_classes=10):
                super(ResNet, self).__init__()
                self.inchannel = 64
                self.conv1 = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU()
                )
                self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
                self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
                self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)        
                self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)        
                self.fc = nn.Linear(512, num_classes)
                self.dropout = nn.Dropout()
                
            def make_layer(self, block, channels, num_blocks, stride):
                strides = [stride] + [1] * (num_blocks - 1)
                layers = []
                for stride in strides:
                    layers.append(block(self.inchannel, channels, stride))
                    self.inchannel = channels
                return nn.Sequential(*layers)
            
            def forward(self, x):
                out = self.conv1(x)
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = F.avg_pool2d(out, 4)
                out = out.view(out.size(0), -1)
                out = self.dropout(out)
                out = self.fc(out)
                return out    
        def Model(num_classes):
            return ResNet(ResidualBlock, num_classes)
    
    return Model

def evaluate_per_label_accuracy(model, testloader, device, num_classes=10):
    """
    Evaluate per-label accuracy on CIFAR-10 (original 10 labels, no remapping).

    Args:
        model (nn.Module): Trained model.
        testloader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run evaluation on.
        num_classes (int): Number of classes (default: 10 for CIFAR-10).

    Returns:
        dict: Per-label accuracy {label_index: accuracy_percentage}.
    """
    model.eval()
    with torch.no_grad():
        class_counts = {i: 0 for i in range(num_classes)}
        class_correct = {i: 0 for i in range(num_classes)}

        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            for class_idx in range(num_classes):
                class_mask = (labels == class_idx)
                class_counts[class_idx] += class_mask.sum().item()
                class_correct[class_idx] += (predictions[class_mask] == class_idx).sum().item()
 
        total_samples = sum(class_counts.values())
        total_correct = sum(class_correct.values())
        
        per_label_accuracy = {}
        for class_idx in range(num_classes):
            if class_counts[class_idx] > 0:
                per_label_accuracy[class_idx] = 100 * class_correct[class_idx] / class_counts[class_idx]
            else:
                per_label_accuracy[class_idx] = 0.0

            print(f"Accuracy for Label {class_idx}: {per_label_accuracy[class_idx]:.2f}%")

        overall_accuracy = 100 * total_correct / total_samples if total_samples > 0 else 0
    
    return per_label_accuracy, overall_accuracy

def save_data_to_csv(accuracy_distributions, contribution_distributions, chosen_users_over_time, expected_gradient_magnitude, expected_true_gradient_magnitude, num_users, num_timeframes, args, current_time, start_time, elapsed_time, end_time, num_runs, seeds_for_avg, num_send, Nmax, P_onmin, Rmin, Nkstar, L, max_sigma_l, max_sigma_g, max_G, fw0):
    """
    Save accuracy distributions and contribution distributions to CSV files.

    :param accuracy_distributions: Accuracy distributions data
    :param contribution_distributions: Contribution distributions data
    :param num_users: Number of users in federated learning
    :param num_timeframes: Number of timeframes for simulation
    :param args: Command-line arguments
    :param current_time: Current timestamp for saving results
    :param start_time: Start time of the run
    :param elapsed_time: Elapsed time of the run
    :param end_time: End time of the run
    :param num_runs: Number of simulation runs
    :param seeds_for_avg: Random seeds used for averaging results
    :param num_send: Average gradients sent per round
    """
    save_dir = f"./results10slot1mem_{current_time}"
    os.makedirs(save_dir, exist_ok=True)

    # Save final results
    final_results = []
    for run in range(num_runs):
        for seed_index, seed in enumerate(seeds_for_avg):
            for timeframe in range(num_timeframes):
                final_results.append({
                    'Run': run,
                    'Seed': seed,
                    'Timeframe': timeframe + 1,
                    'Accuracy': accuracy_distributions[run][seed_index][timeframe],
                })

    contribution_data = []    
    for user in range(num_users):
        for run in range(num_runs):
            for seed_index, seed in enumerate(seeds_for_avg):
                contribution_data.append({
                    'Run': run,
                    'Seed': seed,
                    'User': user,
                    'Contribution': contribution_distributions[run][seed_index][user],
                })

    expected_gradient_data = []
    for user in range(num_users):
        for run in range(num_runs):
            for seed_index, seed in enumerate(seeds_for_avg):
                expected_gradient_data.append({
                    'Run': run,
                    'Seed': seed,
                    'User': user,
                    'ExpectedGradientMagnitude': expected_gradient_magnitude[run][seed_index][user],
                })
    
    expected_gradient_data = []
    for user in range(num_users):
        for run in range(num_runs):
            for seed_index, seed in enumerate(seeds_for_avg):
                expected_gradient_data.append({
                    'Run': run,
                    'Seed': seed,
                    'User': user,
                    'ExpectedGradientMagnitude': expected_gradient_magnitude[run][seed_index][user],
                })

    expected_true_gradient_data = []
    for timeframe in range(num_timeframes):
        for run in range(num_runs):
            for seed_index, seed in enumerate(seeds_for_avg):
                expected_true_gradient_data.append({
                    'Run': run,
                    'Seed': seed,
                    'Timeframe': timeframe + 1,
                    'ExpectedTrueGradientMagnitude': expected_true_gradient_magnitude[run][seed_index][timeframe],
                })


    chosen_users_over_time_data = []
    for run in range(num_runs):
        for seed_index, seed in enumerate(seeds_for_avg):
            for timeframe in range(num_timeframes):
                for user in range(num_users):
                    chosen_users_over_time_data.append({
                        'Run': run,
                        'Seed': seed,
                        'Timeframe': timeframe + 1,
                        'User': user,
                        'ChosenUser': chosen_users_over_time[run][seed_index][timeframe][user],
                    })

    final_results_df = pd.DataFrame(final_results)
    contribution_data_df = pd.DataFrame(contribution_data)
    expected_gradient_data_df = pd.DataFrame(expected_gradient_data)
    expected_true_gradient_data_df = pd.DataFrame(expected_true_gradient_data)
    chosen_users_over_time_df = pd.DataFrame(chosen_users_over_time_data)
    file_path = os.path.join(save_dir, 'final_results.csv')
    final_results_df.to_csv(file_path, index=False)
    contribution_file_path = os.path.join(save_dir, 'contribution_data.csv')
    contribution_data_df.to_csv(contribution_file_path, index=False)
    chosen_users_file_path = os.path.join(save_dir, 'chosen_users_over_time.csv')
    chosen_users_over_time_df.to_csv(chosen_users_file_path, index=False)
    expected_gradient_file_path = os.path.join(save_dir, 'expected_gradient_magnitude.csv')
    expected_gradient_data_df.to_csv(expected_gradient_file_path, index=False)
    expected_true_gradient_file_path = os.path.join(save_dir, 'expected_true_gradient_magnitude.csv')
    expected_true_gradient_data_df.to_csv(expected_true_gradient_file_path, index=False)
    print(f"Final results saved to: {file_path}")
    print(f"Contribution data saved to: {contribution_file_path}")
    print(f"Chosen users over time data saved to: {chosen_users_file_path}")
    print(f"Expected gradient magnitude data saved to: {expected_gradient_file_path}")
    print(f"Expected true gradient magnitude data saved to: {expected_true_gradient_file_path}")
    # Save correctly received packets statistics to CSV

    # Save run summary
    summary_content = (
        f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n"
        f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n"
        f"Elapsed Time: {elapsed_time:.2f} seconds\n"
        f"Arguments: {vars(args)}\n"
        f"Average gradient per round: {num_send/num_timeframes}\n"
        f"Nmax: {Nmax}\n"
        f"P_onmin: {P_onmin}\n"
        f"Rmin: {Rmin}\n"
        f"Nkstar: {Nkstar}\n"
        f"Lipschitz Constant (L): {L:.4f}\n"
        f"Local Variance Bound (sigma_l): {max_sigma_l:.4f}\n"
        f"Global Variance Bound (sigma_g): {max_sigma_g:.4f}\n"
        f"Gradient Magnitude Bound (G): {max_G:.4f}\n"
        f"Initial Global Loss f(w^0): {fw0:.6f}\n"
    )

    summary_file_path = os.path.join(save_dir, 'run_summary.txt')
    with open(summary_file_path, 'w') as summary_file:
        summary_file.write(summary_content)

    print(f"Run summary saved to: {summary_file_path}")

    import torch

def calculate_lipschitz_constant(TrainSetUsers, device="cpu"):
    """
    Calculates the strict global Lipschitz constant L for Logistic Regression 
    across a federated dataset.
    """
    L_max = 0.0
    
    for user_idx, user_dataloader in enumerate(TrainSetUsers):
        features = []
        
        for images, _ in user_dataloader:
            features.append(images.view(images.size(0), -1))
            
        X_u = torch.cat(features, dim=0).to(device)
        N_u = X_u.size(0) # Number of local samples
        
        XT_X = torch.matmul(X_u.T, X_u) / N_u
        
        eigenvalues = torch.linalg.eigvalsh(XT_X)
        lambda_max = eigenvalues.max().item()
        
        L_u = 0.25 * lambda_max
        
        if L_u > L_max:
            L_max = L_u
            
    print(f"--> Global Lipschitz Constant (L): {L_max:.4f}")
    return L_max

def calculate_gradient_difference(w_before, w_after):
    return [w_after[k] - w_before[k] for k in range(len(w_after))]

def calculate_gradient_magnitude(gradient_diff):
    return sum(torch.sum(g**2) for g in gradient_diff).item()

def calculate_whole_gradient(user_id, device, w_current, model, WholeTrainSetUsers, epochs, criterion):
    # Reset model weights to the initial weights before each user's local training
    with torch.no_grad():
        for param, saved in zip(model.parameters(), w_current):
            param.copy_(saved) 
    torch.cuda.empty_cache()

    # Retrieve the user's training data (combined from all memory cells)
    Wholetrainloader = WholeTrainSetUsers[user_id]
    local_optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    
    for epoch in range(epochs):
        for image, label in Wholetrainloader:
            local_optimizer.zero_grad(set_to_none=True)     
            image, label = image.to(device), label.to(device)  
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            local_optimizer.step()
            torch.cuda.empty_cache()

    w_new = [param.data.clone().to(device) for param in model.parameters()]
    gradient_diff = calculate_gradient_difference(w_current, w_new)

    return gradient_diff

def calculate_true_gradient(device, w_current, model, WholeTrainSet, epochs, criterion):
    # Reset model weights to the initial weights before each user's local training
    with torch.no_grad():
        for param, saved in zip(model.parameters(), w_current):
            param.copy_(saved) 
    torch.cuda.empty_cache()

    # Retrieve the user's training data (combined from all memory cells)
    Wholetrainloader = WholeTrainSet
    local_optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    
    for epoch in range(epochs):
        for image, label in Wholetrainloader:
            local_optimizer.zero_grad(set_to_none=True)     
            image, label = image.to(device), label.to(device)  
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            local_optimizer.step()
            torch.cuda.empty_cache()

    w_new = [param.data.clone().to(device) for param in model.parameters()]
    gradient_diff = calculate_gradient_difference(w_current, w_new)

    return gradient_diff

def calculate_constants(num_users, device, w_global, model, TrainSetUsers, WholeTrainSetUsers, WholeTrainSet, epochs, optimizer, criterion):
    
    max_sigma_l = 0.0
    max_sigma_g = 0.0
    max_G = 0.0

    for user_id in range(num_users):
        print(user_id, end=' ')
        cur_sigma_l = 0.0
        iii = 0

        # Reset model weights to the initial weights before each user's local training
        with torch.no_grad():
            for param, saved in zip(model.parameters(), w_global):
                param.copy_(saved) 
        torch.cuda.empty_cache()

        model2 = copy.deepcopy(model).to(device)

        # Retrieve the user's training data (combined from all memory cells)
        trainloader = TrainSetUsers[user_id]
        
        for epoch in range(epochs):
            for image, label in trainloader:
                optimizer.zero_grad(set_to_none=True)     
                image, label = image.to(device), label.to(device)  
                output = model(image)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
                
                w_new = [param.data.clone().to(device) for param in model.parameters()]
                
                gradient_diff = calculate_gradient_difference(w_global, w_new)
                whole_gradient_diff = calculate_whole_gradient(user_id, device, w_global, model2, WholeTrainSetUsers, epochs, criterion)
                true_gradient_diff = calculate_true_gradient(device, w_global, model2, WholeTrainSet, epochs, criterion)
                
                gradient_magnitude = calculate_gradient_magnitude(gradient_diff)
                varience = calculate_gradient_magnitude(calculate_gradient_difference(gradient_diff, whole_gradient_diff))
                whole_true_diff_magnitude = calculate_gradient_magnitude(calculate_gradient_difference(whole_gradient_diff, true_gradient_diff))

                if gradient_magnitude > max_G:
                    max_G = gradient_magnitude
                
                cur_sigma_l += varience

                if whole_true_diff_magnitude > max_sigma_g:
                    max_sigma_g = whole_true_diff_magnitude

                w_global = w_new
                iii += 1
        sigma_l = cur_sigma_l / (iii+1)
        if sigma_l > max_sigma_l:
            max_sigma_l = sigma_l

    print(f"--> Local Variance Bound (sigma_l): {max_sigma_l:.4f}")
    print(f"--> Global Variance Bound (sigma_g): {max_sigma_g:.4f}")
    print(f"--> Gradient Magnitude Bound (G): {max_G:.4f}")
    return max_sigma_l, max_sigma_g, max_G


    
def calculate_init_loss(model, WholeTrainSet, criterion, device):
    """
    Calculates the f(w^0) 
    """
    print("\nCalculating f(w^0)")
    
    # ---------------------------------------------------------
    # 1. Calculate Initial Global Loss: f(w^0)
    # ---------------------------------------------------------
    model.eval()
    initial_loss_sum = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in WholeTrainSet:
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Multiply by batch size to get the total sum of losses
            initial_loss_sum += loss.item() * images.size(0)
            total_samples += images.size(0)
            
    f_w0 = initial_loss_sum / total_samples
    print(f"--> Initial Global Loss f(w^0): {f_w0:.6f}")
    return f_w0