from collections.abc import Mapping
import torch
from utils.args import args
import numpy as np
from collections import Counter
import torch.nn.functional as F

def get_domains_and_labels(arguments):    
    if arguments.dataset.name == 'CalD3rMenD3s':
        num_class = 7
        valid_labels = [i for i in range(num_class)]
        
    return num_class, valid_labels


class Accuracy(object):
    """Computes and stores the average and current value of different top-k accuracies from the outputs and labels"""
    #?Top-K Accuracy is a generalization of top-1 accuracy. It considers the prediction correct if the true label is among the top K predicted classes. 
    #? For example, top-5 accuracy means that the true label is considered correctly predicted if it is among the model's top 5 predicted classes. 

    def __init__(self, topk=(1,5)):
        assert len(topk) > 0
        self.topk = topk
        self.num_classes, _ = get_domains_and_labels(args)
        self.avg, self.val, self.sum, self.count, self.correct, self.total = None, None, None, None, None, None
        self.reset()

    def reset(self):
        ''' reset the current value, average, sum, and count of accuracies for each top-k scoring. '''
        self.val = {tk: 0 for tk in self.topk}
        self.avg = {tk: 0 for tk in self.topk}
        self.sum = {tk: 0 for tk in self.topk}
        self.count = {tk: 0 for tk in self.topk}
        self.correct = list(0 for _ in range(self.num_classes)) #?  list to store number of correct predictions per class
        self.total = list(0 for _ in range(self.num_classes)) #? list to store number of samples per class

    def update(self, outputs, labels):
        '''updates the accuracy metrics with new batch '''

        if outputs.dim() == 1: #? if the output is a single element in the final batch, unsqueeze it to have a batch of size 1
            outputs = outputs.unsqueeze(0)
            labels = labels.unsqueeze(0)
            
        batch = labels.size(0)

        for i_tk, top_k in enumerate(self.topk): #? for each top-k scoring , update the current value, sum, count, average, per-class correct and total counts. 
            
            if i_tk == 0: #? if it is the first top-k scoring, coompute also per_class accuracy
                res, class_correct, class_total = self.accuracy(outputs, labels,  topk=top_k, perclass_acc=True)
            else:
                res, _, _ = self.accuracy(outputs, labels, topk=top_k, perclass_acc=False)

            self.val[top_k] = res
            self.sum[top_k] += res * batch
            self.count[top_k] += batch
            self.avg[top_k] = self.sum[top_k] / self.count[top_k]

        for i in range(0, self.num_classes):
            self.correct[i] += class_correct[i]
            self.total[i] += class_total[i]

    def accuracy(self, output, labels, topk, perclass_acc=False):
        """
        Computes the top-k accuracy for the given outputs and labelss
        output: torch.Tensor -> the predictions #? [batch_size, num_classes]
        labels: torch.Tensor -> ground truth labels #? [batch_size]
        perclass_acc -> bool, True if you want to compute also the top-1 accuracy per class
        """
        batch_size = labels.size(0)
  
        #? .topk() returns the positions of the k largest elements of the given input tensor along given dimension. (num_classes in this case)
        #? "True, True" specify that the elements should be sorted in descending order and that we want to sort the data itself, not a copy.
        _, pred = output.topk(topk, 1, True, True) #? [batch_size, topk]
        pred = pred.t() #? transpose [topk, batch_size]
        
        #? pred.eq() returns a tensor of the same size as input with True where the elements are equal and False where they are not.
        #? labels.view(1, -1).expand_as(pred) -> reshapes the labels tensor to have the same shape as the pred tensor
        check1 = labels.view(1, -1) #[1, batch_size]
        check2 = labels.view(1, -1).expand_as(pred) #[topk, batch_size]
        
        correct = pred.eq(labels.view(1, -1).expand_as(pred)) 
        correct_k = correct[:topk].reshape(-1).to(torch.float32).sum(0) #number of correct predictions in the batch for the top-k scoring
        
        res = float(correct_k.mul_(100.0 / batch_size)) #transform to percentage
        class_correct, class_total = None, None
        if perclass_acc:
            # getting also top1 accuracy per class
            class_correct, class_total = self.accuracy_per_class(correct[:1].view(-1), labels)
            
        return res, class_correct, class_total

    def accuracy_per_class(self, correct, labels):
        """
        function to compute the accuracy per class
        correct -> (batch, bool): vector which, for each element of the batch, contains True/False depending on if
                                  the element in a specific poisition was correctly classified or not
        labels -> (batch, label): vector containing the ground truth for each element
        """
        class_correct = list(0. for _ in range(0, self.num_classes))
        class_total = list(0. for _ in range(0, self.num_classes))
        for i in range(0, labels.size(0)):
            class_label = labels[i].item()
            class_correct[class_label] += correct[i].item()
            class_total[class_label] += 1
            
        return class_correct, class_total


class LossMeter(object):
    """Computes and stores the average and current value
       Used for loss update  
    """
    def __init__(self):
        self.reset()
        self.val, self.acc, self.avg, self.sum, self.count = 0, 0, 0, 0, 0

    def reset(self):
        self.val = 0
        self.acc = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.acc += val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_class_weights(train_loader, norm=False):
    """This function computes weights for each class used in Weighted losses
    """
   # Step 1: Collect all labels from the DataLoader
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.tolist())  # Assuming labels are tensors, convert to list and extend
    
    # Step 2: Count class frequencies
    class_counts = Counter(all_labels)

    # Step 3: Compute class weights
    total_samples = len(all_labels)
    num_classes = len(class_counts)
    class_weights = [total_samples / (class_counts[i] * num_classes) for i in range(num_classes)]

    # Step 4: Normalize weights (optional)
    if norm: 
        sum_weights = sum(class_weights)
        class_weights = [w / sum_weights for w in class_weights]

    return torch.FloatTensor(class_weights)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
        
    def generate_cam(self, input_tensor, target_class=None):
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Backpropagate to get gradients with respect to the target class
        class_loss = output[:, target_class].sum()
        class_loss.backward(retain_graph=True)
        
        # Get the gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        # Compute the weight of each feature map
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        
        # Compute the Grad-CAM
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize the CAM
        cam = cam - cam.min()
        cam = cam / cam.max()
        cam = cam.squeeze().cpu().detach().numpy()

        return cam
 
 
def pformat_dict(d, indent=0):
    fstr = ""
    for key, value in d.items():
        fstr += '\n' + '  ' * indent + str(key) + ":"
        if isinstance(value, Mapping):
            fstr += pformat_dict(value, indent+1)
        else:
            fstr += ' ' + str(value)
    return fstr


