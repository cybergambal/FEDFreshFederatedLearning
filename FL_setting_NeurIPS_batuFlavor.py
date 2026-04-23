import csv
import os
import random
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import time
import heapq
class FederatedLearning:
    def __init__(self, mode, num_users, device, 
                    cos_similarity, model, TrainSetUsers, WholeTrainSet, epochs, optimizer, criteron, fraction, 
                    testloader, learning_rate_server, train_mode, keepProbAvail, keepProbNotAvail, 
                    bufferLimit, theta_inner, unit_gradients, adam, temp, cos_similarity_type):
        
        #Arguements
        self.learning_rate_server = learning_rate_server
        self.epochs = epochs
        self.num_users = num_users
        self.fraction = fraction
        self.mode = mode
        self.cos_similarity = cos_similarity
        self.bufferLimit = bufferLimit
        self.theta_inner = theta_inner
        self.train_mode = train_mode
        self.unit_gradients = unit_gradients
        self.adam = adam

        #Device
        self.device = device 

        #Weights in each user 
        self.w_user = [[param.data.clone().to("cpu") for param in model.parameters()] for _ in range(num_users)]

        #Global Weights
        self.w_global = [param.data.clone().to(self.device) for param in model.parameters()]

        #Sparse gradients of users 
        self.sparse_gradient = [[torch.zeros_like(param).to("cpu") for param in model.parameters()] for _ in range(num_users)]

        #Aggregation buffer
        self.sum_terms = [torch.zeros_like(param).to(self.device) for param in self.w_global]

        #Training components 
        self.model = model
        self.optimizer = optimizer
        self.criteron = criteron
        self.TrainSetUsers = TrainSetUsers 
        self.WholeTrainSet = WholeTrainSet
        self.testloader = testloader

        #Intermittent user model
        self.keepProbAvail = keepProbAvail
        self.keepProbNotAvail = keepProbNotAvail
        self.intermittentStateOneHot = np.array([1 if (1-self.keepProbNotAvail[u])/(2-self.keepProbAvail[u]-self.keepProbNotAvail[u]) > random.random() else 0 for u in range(num_users)])
        self.intermittentUsers = np.where(self.intermittentStateOneHot)

        #Age based variables
        self.UserAgeUL = torch.zeros(self.num_users, 1).to("cpu")
        self.UserAgeDL = torch.ones(self.num_users, 1).to("cpu") 
        self.UserAgeMemory = torch.zeros(self.num_users, 1).to("cpu")
        self.allOnes = torch.ones(self.num_users, 1).to("cpu")


        #Inner Product Test variables 
        self.bufferSize = 0
        self.userListUL = set(range(self.num_users))
        self.setAllUsers = set(range(self.num_users))
        self.nu_orthogonal = 5.67 #tan(80)

        #Policy calculation
        self.Nmax = 0.0
        self.P_onmin = 0.0
        self.Rmin = 0.0
        self.Nkstar = 0.0
        self.temperature = temp
        self.pi = self.calculate_policy()
        
        # Tracking variables
        self.contribution = np.zeros((self.num_users, 1))
        self.expected_gradient_magnitude = np.zeros((self.num_users, 1))
        self.num_send = 0

        #Cosine similarity variables
        self.lastGradient = [torch.zeros_like(param).to(self.device) for param in self.w_global]
        self.lastGradient = torch.cat([g.view(-1) for g in self.lastGradient]).t()
        self.cosine_similarity_type = cos_similarity_type

        # Adam parameters
        if self.adam: 
            self.beta1 = 0.9
            self.beta2 = 0.99
            self.tau = 1e-3 

            self.adamMomentum = [torch.zeros_like(param).to(self.device) for param in self.w_global]
            self.adamVariance = [torch.full_like(param, self.tau**2).to(self.device) for param in self.w_global]

    def log_gamma_step(self, timeframe, lam, p_unif_np, p_eff_np, log_path="gamma_log.csv"):
        """Append one (timeframe, lam, inner_unif, inner_eff, grad_f_sq) row to the CSV.

        Measures the gradient-alignment ratio from Assumption 5 at the current
        w_global using a true full-batch forward/backward per user.  Read-only:
        no optimizer step, no weight update, no BN running-stat update.
        """
        self.model.eval()

        # Sync model weights to current global weights
        with torch.no_grad():
            for param, saved in zip(self.model.parameters(), self.w_global):
                param.copy_(saved)

        # Compute each user's full-batch gradient at w_global
        user_grads = []
        for user_id in range(self.num_users):
            self.optimizer.zero_grad(set_to_none=True)
            total_loss = torch.tensor(0.0, device=self.device)
            total_samples = 0
            for image, label in self.TrainSetUsers[user_id]:
                image, label = image.to(self.device), label.to(self.device)
                out = self.model(image)
                loss = self.criteron(out, label) * image.size(0)
                total_loss = total_loss + loss
                total_samples += image.size(0)
            (total_loss / max(total_samples, 1)).backward()
            flat = torch.cat([
                p.grad.detach().view(-1) if p.grad is not None
                else torch.zeros(p.numel(), device=self.device)
                for p in self.model.parameters()
            ]).cpu()
            user_grads.append(flat)

        user_grads = torch.stack(user_grads)  # [U, D]
        dtype = user_grads.dtype
        p_unif_t = torch.as_tensor(p_unif_np, dtype=dtype)
        p_eff_t = torch.as_tensor(p_eff_np, dtype=dtype)

        # Global gradient = uniform average; policy-weighted aggregates
        grad_f = user_grads.mean(dim=0)
        grad_f_sq = float(torch.dot(grad_f, grad_f).item())

        agg_unif = (p_unif_t.unsqueeze(1) * user_grads).sum(dim=0)
        agg_eff = (p_eff_t.unsqueeze(1) * user_grads).sum(dim=0)
        inner_unif = float(torch.dot(agg_unif, grad_f).item())
        inner_eff = float(torch.dot(agg_eff, grad_f).item())

        write_header = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timeframe", "lam", "inner_unif", "inner_eff", "grad_f_sq"])
            w.writerow([timeframe, lam, inner_unif, inner_eff, grad_f_sq])

        self.model.train()

    def calculate_gradient_magnitude(self, gradient_diff):
        return sum(torch.sum(g**2) for g in gradient_diff).item()

    def lp_cosine_similarity(self, x: torch.Tensor, y: torch.Tensor, p: int = 2) -> float:
        """
        Compute the Lp cosine similarity between two flattened gradient vectors.
    
        Args:
            x (torch.Tensor): 1D tensor.
            y (torch.Tensor): 1D tensor.
            p (int): Norm degree (e.g., 2 for L2).
    
        Returns:
            float: The Lp cosine similarity.
        """
        norm_x = torch.norm(x, p=p)
        norm_y = torch.norm(y, p=p)
        norm_x_plus_y_sq = torch.norm(x + y, p=p) ** 2
        norm_x_sq = norm_x ** 2
        norm_y_sq = norm_y ** 2

        numerator = 0.5 * (norm_x_plus_y_sq - norm_x_sq - norm_y_sq)
        denominator = norm_x * norm_y + 1e-12  # avoid division by zero

        return (numerator / denominator).item()
    
    def cosine_similarity_policy(self) -> List[int]:

        """ Select user with highest cosine similarity to last gradient """

        valList = []
        userList = []

        for user in self.intermittentUsers:
            user_grad_vector = torch.cat([g.view(-1) for g in self.sparse_gradient[user]]).to(self.device)
            cos_sim = self.lp_cosine_similarity(user_grad_vector, self.lastGradient, p = self.cos_similarity)
            valList.append(cos_sim)
            userList.append(user)
            print(f"Cosine Similarity for user {user}: {cos_sim}")
        
        chosen_list = heapq.nlargest(self.bufferLimit, zip(valList, userList), key=lambda x: x[0]) if self.cosine_similarity_type else heapq.nsmallest(self.bufferLimit, zip(valList, userList), key=lambda x: x[0])

        chosen_list = [user for val, user in chosen_list]

        return chosen_list

    def calculate_true_gradient_magnitude(self):
        # Reset model weights to the initial weights before each user's local training
        with torch.no_grad():
            for param, saved in zip(self.model.parameters(), self.w_global):
                param.copy_(saved) 
        torch.cuda.empty_cache()

        # Retrieve the user's training data (combined from all memory cells)
        Wholetrainloader = self.WholeTrainSet
        local_optimizer = torch.optim.SGD(self.model.parameters(), lr=1.0)
        
        for epoch in range(self.epochs):
            for image, label in Wholetrainloader:
                local_optimizer.zero_grad(set_to_none=True)     
                image, label = image.to(self.device), label.to(self.device)  
                output = self.model(image)
                loss = self.criteron(output, label)
                loss.backward()
                local_optimizer.step()
                torch.cuda.empty_cache()

        w_new = [param.data.clone().to(self.device) for param in self.model.parameters()]
        gradient_diff = self.calculate_gradient_difference(self.w_global, w_new)
        grad_mag = self.calculate_gradient_magnitude(gradient_diff)
        print(f"True Gradient Magnitude: {grad_mag}")

        return grad_mag
    
    def calculate_policy(self):
        pi = np.zeros((self.num_users))
        r = np.zeros((self.num_users)) 
        pon = np.zeros((self.num_users))
        denums = np.zeros((self.num_users))
        pi_cont = np.hstack((np.zeros((self.num_users//2)), np.ones((self.num_users - self.num_users//2))))
        n_u_array = np.zeros((self.num_users))

        for iii in range(self.num_users):
            P10 = 1 - self.keepProbAvail[iii]
            P01 = 1 - self.keepProbNotAvail[iii]
            
            # Numerator
            term1 = (1 - P10)
            term2 = (P10 * P01) / (1 - P01)
            term3 = (P10 * P01) / ((1 - P01) ** 2) * np.log(P01)
            numerator = term1 - term2 - term3

            n_u_array[iii] = numerator
            
            # Denominator
            denominator = 1 + P10 / P01
            denums[iii] = denominator
            
            r[iii] = numerator / denominator
            pon[iii] = P01/(P01 + P10)

        self.Rmin = np.min(r)
        self.Nmax = np.max(n_u_array)
        self.P_onmin = np.min(pon)
        self.Nkstar = self.Nmax

        inverseSum = np.sum(r**(-1))
        pi = (r**(-1) / inverseSum)

        # SortFunc = lambda a : a[1]
        # rTemp = list(enumerate(list(r)))
        # rTemp.sort(key=SortFunc, reverse=True)
        # cap = self.bufferLimit
        # for iii in range(self.num_users):
        #     user = rTemp[iii]

        #     if user[1] < cap:
        #         pi_cont[user[0]] = 1
        #         cap -= user[1]
        #     else:
        #         pi_cont[user[0]] = cap/user[1]
        #         break 

        pi_cont = pi_cont / np.dot(pon, pi_cont) * self.bufferLimit
        
        print(f"pi_cont: {pi_cont}")

        pi = pi / np.dot(pon, pi) * self.bufferLimit
        print(f"pi: {pi}")

        pi = pi_cont * (1 - self.temperature) + pi * self.temperature
        print(f"Overall pi: {pi}")

        return pi

    def innerProductTest(self):
        """" Inner Product Test from paper "" """
        if self.bufferSize == 0:
            return False
        choosenUsers = self.setAllUsers.difference(self.userListUL)
        
        global_grad_vector = torch.cat([(g/self.bufferSize).view(-1) for g in self.sum_terms])
        gradMag = torch.dot(global_grad_vector, global_grad_vector)
        print(self.bufferSize)
        varEst = 0
        for user in choosenUsers:
            user_grad_vector = torch.cat([(g/self.UserAgeMemory[user]).view(-1) for g in self.sparse_gradient[user]]).t()
            accInner = torch.dot(user_grad_vector, global_grad_vector)
            print(accInner/torch.norm(user_grad_vector)/torch.norm(global_grad_vector))
            varEst = varEst + torch.square(accInner-gradMag)
        varEst = varEst/max(1, self.bufferSize-1)
        
        conLHS = varEst/self.bufferSize
        conRHS = torch.square(self.theta_inner*gradMag)
        print("Inner Product Test:", conLHS, "<=", conRHS)
        check = conLHS <= conRHS
        return check 

    def orthogonalityTest(self):
        """" Inner Product Test from paper "" """
        if self.bufferSize == 0:
            return False
        choosenUsers = self.setAllUsers.difference(self.userListUL)

        global_grad_vector = torch.cat([(g/self.bufferSize).view(-1) for g in self.sum_terms])
        gradMag = torch.dot(global_grad_vector, global_grad_vector)

        orthTest = 0
        for user in choosenUsers:
            user_grad_vector = torch.cat([g.view(-1) for g in self.sparse_gradient[user]])
            accInner = torch.dot(user_grad_vector, global_grad_vector)
            grad = user_grad_vector - accInner/gradMag*global_grad_vector
            orthTest = orthTest + torch.dot(grad, grad)
        
        
        conLHS = orthTest/(max(1, self.bufferSize-1)*self.bufferSize)
        conRHS = (self.nu_orthogonal*self.nu_orthogonal)*gradMag
        print("Orthoganality Test:", conLHS, "<=", conRHS)

        check = conLHS <= conRHS 
        return check

    def stepState(self):
        for iii in range(self.num_users):
            if (self.intermittentStateOneHot[iii]):
                self.intermittentStateOneHot[iii] = self.intermittentStateOneHot[iii] if self.keepProbAvail[iii] > random.random() else 1-self.intermittentStateOneHot[iii]
            else:
                self.intermittentStateOneHot[iii] = self.intermittentStateOneHot[iii] if self.keepProbNotAvail[iii] > random.random() else 1-self.intermittentStateOneHot[iii]
        self.intermittentUsers = np.where(self.intermittentStateOneHot)[0]

    # Calculate gradient difference between two sets of weights
    def calculate_gradient_difference(self, w_before, w_after):
        return [w_after[k] - w_before[k] for k in range(len(w_after))]
    
    # Sparsify the model weights
    def top_k_sparsificate_model_weights(self, weights, fraction):
        flat_weights = torch.cat([w.view(-1) for w in weights])
        threshold_value = torch.quantile(torch.abs(flat_weights), 1 - fraction)
        new_weights = []
        for w in weights:
            mask = torch.abs(w) >= threshold_value
            new_weights.append(w * mask.float())
        return new_weights
    

    def train_users(self, list_users):
        for user_id in list_users:

            # Reset model weights to the initial weights before each user's local training
            model = [param.data.clone().to(self.device) for param in self.w_user[user_id]]
            with torch.no_grad():
                for param, saved in zip(self.model.parameters(), model):
                    param.copy_(saved) 
            torch.cuda.empty_cache()

            # Retrieve the user's training data (combined from all memory cells)
            trainloader = self.TrainSetUsers[user_id]
            
            if self.train_mode == "MNIST":
                for epoch in range(self.epochs):
                    for image, label in trainloader:
                        self.optimizer.zero_grad(set_to_none=True)     
                        image, label = image.to(self.device), label.to(self.device)  
                        output = self.model(image)
                        loss = self.criteron(output, label)
                        loss.backward()
                        self.optimizer.step()
                        torch.cuda.empty_cache()
            else: 
                for epoch in range(self.epochs): 
                    for image, label in trainloader:
                        self.optimizer.zero_grad(set_to_none=True)
                        image, label = image.to(self.device), label.to(self.device)  
                        output = self.model(image)
                        loss = self.criteron(output, label)
                        loss.backward()

                        self.optimizer.step()
        
            w_new = [param.data.clone().to(self.device) for param in self.model.parameters()]
            gradient_diff = self.calculate_gradient_difference(model, w_new)
            sparse_gradient = self.top_k_sparsificate_model_weights(gradient_diff, self.fraction[0]) 
            self.sparse_gradient[user_id] = [sg.to("cpu") for sg in sparse_gradient]

    def aggregate_gradients(self, tempUserAgeDL):

        # Normalize gradients if unit_gradients is set
        if self.unit_gradients:
            acc = 0 
            for user in self.selected_users_UL:
                norm = np.sqrt(sum([torch.sum(g**2).item() for g in self.sparse_gradient[user]]))
                if norm > 0:
                    self.sparse_gradient[user] = [g / norm for g in self.sparse_gradient[user]]
                
                norm = np.sqrt(sum([torch.sum(g**2).item() for g in self.sparse_gradient[user]]))
                print(f"Norm of sparse gradient for user {user}: {norm}")
    
        #Sum of trained gradients
        self.sum_terms = [torch.zeros_like(param).to(self.device) for param in self.w_global]
        for user in self.selected_users_UL:
            self.UserAgeUL[user] = 0
            self.contribution[user] += np.sqrt(sum([torch.sum((g/tempUserAgeDL[user].item())**2).item() for g in self.sparse_gradient[user]]))
            temp_gradient = [sg.to(self.device) for sg in self.sparse_gradient[user]]
            self.expected_gradient_magnitude[user] += np.sqrt(sum([torch.sum(g**2).item() for g in temp_gradient]))
            self.sum_terms = [self.sum_terms[j] + temp_gradient[j]/(tempUserAgeDL[user]) for j in range(len(self.sum_terms))] 
        
        if self.adam:
            # Adam update
            self.adamMomentum = [self.beta1 * m + (1 - self.beta1) * (s / len(self.selected_users_UL)) for m, s in zip(self.adamMomentum, self.sum_terms)]
            self.adamVariance = [self.beta2 * v + (1 - self.beta2) * ((s / len(self.selected_users_UL)) ** 2) for v, s in zip(self.adamVariance, self.sum_terms)]

            self.lastGradient = [ self.learning_rate_server * self.adamMomentum[j] / (torch.sqrt(self.adamVariance[j]) + self.tau) for j in range(len(self.sum_terms))]
            
            self.lastGradient = torch.cat([g.view(-1) for g in self.lastGradient]).t()
            
            # Update global model
            self.w_global = [self.w_global[j] + self.learning_rate_server * self.adamMomentum[j] / (torch.sqrt(self.adamVariance[j]) + self.tau) for j in range(len(self.sum_terms))] 
        else:
            self.lastGradient = [s / len(self.selected_users_UL) for s in self.sum_terms]
            self.lastGradient = torch.cat([g.view(-1) for g in self.lastGradient]).t()
            
            # Update global model
            self.w_global = [self.w_global[j] + self.learning_rate_server * self.sum_terms[j]/len(self.selected_users_UL) for j in range(len(self.sum_terms))] 
        
    def simulate_async_Asymp_EI(self, run, seed_index, timeframe):
        """Handles both Slotted ALOHA and standard user processing."""

        #New Available Users
        self.stepState()
        if (len(self.intermittentUsers) == 0):
            print("No users available passing")
            return self.w_global
        print(f"Available Users = {self.intermittentUsers}")

        #Choose available users according to their p_u
        tempPi = self.pi[self.intermittentUsers].flatten()            
        bernoulli_flips = np.random.rand(len(self.intermittentUsers)) < tempPi
        self.selected_users_UL = self.intermittentUsers[bernoulli_flips]
        self.num_send += len(self.selected_users_UL)
        if (len(self.selected_users_UL) == 0):
            print("No user transmits")
            return self.w_global
        print(f"Transmitting Users: {self.selected_users_UL.tolist()}")
        
        #Obtain gradient from users that transmit
        self.train_users(self.selected_users_UL.tolist())
        

        tempUserAgeDL = self.UserAgeDL.clone().to(self.device)

        #Available users get the new global model
        for user in self.intermittentUsers:
            self.w_user[user] = [w.clone() for w in self.w_global]
            self.UserAgeDL[user] = 0

        self.aggregate_gradients(tempUserAgeDL)
        
        self.UserAgeDL = self.UserAgeDL + self.allOnes
        
        return self.w_global
    
    def simulate_async_Asymp_Age(self, run, seed_index, timeframe):
        """Handles both Slotted ALOHA and standard user processing."""

        self.UserAgeUL = self.UserAgeUL + self.allOnes 
        
        #New Available Users
        self.stepState()
        if (len(self.intermittentUsers) == 0):
            print("No users available passing")
            return self.w_global
        print(f"Available Users = {self.intermittentUsers}")

        tempUserAgeUL = self.UserAgeUL[self.intermittentUsers]
        print(f"User Age UL: {tempUserAgeUL.squeeze()}")
        tempUserAgeDL = self.UserAgeDL[self.intermittentUsers] 
        print(f"User Age DL: {tempUserAgeDL.squeeze()}")

        # Calculate age difference and select top-k users
        age_diff = (tempUserAgeUL).squeeze()
        k = min(int(self.bufferLimit), len(self.intermittentUsers))        
        sorted_indices = torch.atleast_1d(torch.argsort(age_diff, descending=True))
        topk_indices = sorted_indices[:k]
        self.selected_users_UL = self.intermittentUsers[topk_indices.cpu().numpy()]
        print(f"Selected User in UL: {self.selected_users_UL}")
        
        #Obtain gradient from users that transmit
        self.train_users(self.selected_users_UL.tolist())

        tempUserAgeDL = self.UserAgeDL.clone().to(self.device)
        
        #Available users get the new global model
        for user in self.intermittentUsers:
            self.w_user[user] = [w.clone() for w in self.w_global]
            self.UserAgeDL[user] = 0

        self.aggregate_gradients(tempUserAgeDL) 

        self.UserAgeDL = self.UserAgeDL + self.allOnes

        return self.w_global
    
    def simulate_async_Asymp_CosSim(self, run, seed_index, timeframe):
        """Handles both Slotted ALOHA and standard user processing."""

        self.UserAgeUL = self.UserAgeUL + self.allOnes 
        
        #New Available Users
        self.stepState()
        if (len(self.intermittentUsers) == 0):
            print("No users available passing")
            return self.w_global
        print(f"Available Users = {self.intermittentUsers}")

        
        self.train_users(self.intermittentUsers.tolist())


        self.selected_users_UL = self.cosine_similarity_policy()

        print(f"Selected User in UL: {self.selected_users_UL}")
        
        #Obtain gradient from users that transmit
        tempUserAgeDL = self.UserAgeDL.clone().to(self.device)
        
        #Available users get the new global model
        for user in self.intermittentUsers:
            self.w_user[user] = [w.clone() for w in self.w_global]
            self.UserAgeDL[user] = 0


        self.aggregate_gradients(tempUserAgeDL) 

        self.UserAgeDL = self.UserAgeDL + self.allOnes

        return self.w_global
    
    def simulate_async_Asymp_random(self, run, seed_index, timeframe):
        """Handles both Slotted ALOHA and standard user processing."""

        self.UserAgeUL = self.UserAgeUL + self.allOnes 
        
        #New Available Users
        self.stepState()
        if (len(self.intermittentUsers) == 0):
            print("No users available passing")
            return self.w_global
        print(f"Available Users = {self.intermittentUsers}")

        idx = torch.randint(0, len(self.intermittentUsers), (self.bufferLimit,)).tolist()

        self.selected_users_UL = [self.intermittentUsers[i] for i in idx]

        self.train_users(self.selected_users_UL)

        print(f"Selected User in UL: {self.selected_users_UL}")
        
        #Obtain gradient from users that transmit
        tempUserAgeDL = self.UserAgeDL.clone().to(self.device)
        
        #Available users get the new global model
        for user in self.intermittentUsers:
            self.w_user[user] = [w.clone() for w in self.w_global]
            self.UserAgeDL[user] = 0


        self.aggregate_gradients(tempUserAgeDL) 

        self.UserAgeDL = self.UserAgeDL + self.allOnes

        return self.w_global
    
    def simulate_async_Asymp_Fresh(self, run, seed_index, timeframe):
        """Handles both Slotted ALOHA and standard user processing."""

        self.UserAgeUL = self.UserAgeUL + self.allOnes 
        
        #New Available Users
        self.stepState()
        if (len(self.intermittentUsers) == 0):
            print("No users available passing")
            return self.w_global
        print(f"Available Users = {self.intermittentUsers}")

        FreshUsers = np.where(self.UserAgeDL[self.intermittentUsers]==1)[0]

        if len(FreshUsers) == 0:
            print("No fresh users available")
            return self.w_global

        k = min(self.bufferLimit, len(FreshUsers))
        idx = torch.randint(0, len(FreshUsers), (k,)).tolist()
        self.selected_users_UL = self.intermittentUsers[FreshUsers[idx]]

        self.train_users(self.selected_users_UL)

        print(f"Selected User in UL: {self.selected_users_UL}")
        
        #Obtain gradient from users that transmit
        tempUserAgeDL = self.UserAgeDL.clone().to(self.device)
        
        #Available users get the new global model
        for user in self.intermittentUsers:
            self.w_user[user] = [w.clone() for w in self.w_global]
            self.UserAgeDL[user] = 0


        self.aggregate_gradients(tempUserAgeDL) 

        self.UserAgeDL = self.UserAgeDL + self.allOnes

        return self.w_global
    
    def simulate_test(self, run, seed_index, timeframe):
        self.train_users(list(range(self.num_users)))
        for user_id in range(self.num_users):
            for user_id2 in range(user_id, self.num_users):
                # Flatten gradients into 1D vectors
                user_grad_vector = torch.cat([g.view(-1) for g in self.sparse_gradient[user_id]])
                global_grad_vector = torch.cat([g.view(-1) for g in self.sparse_gradient[user_id2]])

                # Compute cosine similarity
                lp_cos_val = self.lp_cosine_similarity(user_grad_vector, global_grad_vector, p = self.cos_similarity)
                print(f"Similarity between {user_id} and {user_id2} = {lp_cos_val}")

    def run(self, runNo, seed_index, timeframe):
        """Dispatch based on the FL mode."""
        if self.mode == 'test':
            return self.test(runNo, seed_index, timeframe)
        elif self.mode == 'async_asymp_EI':
            return self.simulate_async_Asymp_EI(runNo, seed_index, timeframe)
        elif self.mode == 'async_asymp_age':
            return self.simulate_async_Asymp_Age(runNo, seed_index, timeframe)
        elif self.mode == 'async_asymp_cossim':
            return self.simulate_async_Asymp_CosSim(runNo, seed_index, timeframe)
        elif self.mode == 'async_asymp_random':
            return self.simulate_async_Asymp_random(runNo, seed_index, timeframe)
        elif self.mode == 'async_asymp_fresh':
            return self.simulate_async_Asymp_Fresh(runNo, seed_index, timeframe)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
 