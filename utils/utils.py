from collections.abc import Mapping
import torch
from utils.args import args
import numpy as np
from collections import Counter
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import cv2
import PIL.Image as Image

def get_domains_and_labels(arguments):    
    if arguments.dataset.name == 'CalD3rMenD3s' or arguments.dataset.name == 'BU3DFE':
        if args.FER6:
            num_class = 6
        else:
            num_class = 7
    if arguments.dataset.name == 'Global':
        num_class = 7
        
    return num_class


class Accuracy(object):
    """Computes and stores the average and current value of different top-k accuracies from the outputs and labels"""
    #?Top-K Accuracy is a generalization of top-1 accuracy. It considers the prediction correct if the true label is among the top K predicted classes. 
    #? For example, top-5 accuracy means that the true label is considered correctly predicted if it is among the model's top 5 predicted classes. 

    def __init__(self, topk=(1,5)):
        assert len(topk) > 0
        self.topk = topk
        self.num_classes = get_domains_and_labels(args)
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
        Computes the top-k accuracy for the given outputs and labels
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


def compute_mean_std(train_subset):
    mean = {'RGB': np.zeros(3), 'DEPTH': np.zeros(3)}
    std = {'RGB': np.zeros(3), 'DEPTH': np.zeros(3)}
    sum_pix = {'RGB': np.zeros(3), 'DEPTH': np.zeros(3)}
    sum_sq_pix = {'RGB': np.zeros(3), 'DEPTH': np.zeros(3)}
    n_pix = {'RGB': 0, 'DEPTH': 0}

    for ann_sample in train_subset:
        for m in ['RGB', 'DEPTH']:
            norm_value = 255.0 if m == 'RGB' else 9785.0 
            img = np.array(ann_sample[0][m]) / norm_value  # Normalize to [0, 1]
            
            if m == 'DEPTH':
                #convert from 1 channel to 3 channels
                img = np.expand_dims(img, axis=-1)
                img = np.repeat(img, 3, axis=-1)
             
        
            #Resize images to input size of the model you will use
            img = cv2.resize(img, (260, 260), interpolation=cv2.INTER_LINEAR)
            
            # Update sums
            sum_pix[m] += np.sum(img, axis=(0, 1)) #sum all pixels in the image, separately for each channel (black pixels are 0)
            sum_sq_pix[m] += np.sum(img ** 2, axis=(0, 1))
            
            # Create a mask to maintain only pixels where all three channels are below the threshold
            mask = (img[:, :, 0] > 0) | (img[:, :, 1] >  0) | (img[:, :, 2] > 0)
            #mask off 0 values (black pixels) in the frame, from each channel
            img = img[mask]
            n_pix[m] += mask.sum() #mask converts into a 2D array

    for m in ['RGB', 'DEPTH']:
        mean[m] = sum_pix[m] / n_pix[m]
        std[m] = np.sqrt(sum_sq_pix[m] / n_pix[m] - mean[m] ** 2)
    
    return mean, std


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook the gradients and activations
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def __call__(self, X, target_class):
        img = X['RGB'].unsqueeze(0)  # Add batch dimension
        depth = X['DEPTH'].unsqueeze(0)
        self.model.zero_grad()
        
        # Forward pass
        output, _ = self.model(img, depth)
        
        # Backpropagate to get gradients with respect to the target class score
        output = output.squeeze()
        target_output = output[target_class]
        target_output.backward(retain_graph=True)
        
        # Compute the weight of each feature map
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # Compute the Grad-CAM
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        gradcam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        gradcam = F.relu(gradcam)
        gradcam = F.interpolate(gradcam, size=(X['RGB'].size(1), X['RGB'].size(2)), mode='bilinear', align_corners=False)
        gradcam = gradcam.squeeze().cpu().numpy()
        
        # Normalize the CAM
        gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())

        return gradcam
 
def plot_confusion_matrix(confusion_matrix, fold):
    emotions = {'anger':0, 'disgust':1, 'fear':2, 'happiness':3, 'neutral':4, 'sadness':5, 'surprise':6}
    num_classes = get_domains_and_labels(args)     
    if num_classes == 6:
        if args.dataset.name == 'CalD3rMenD3s':
            emotions = {'anger':0, 'disgust':1, 'fear':2, 'happiness':3, 'neutral':4, 'sadness':5}
        if args.dataset.name == 'BU3DFE': 
            emotions = {'anger':0, 'disgust':1, 'fear':2, 'happiness':3, 'sadness':4, 'surprise':5}
    emotion_labels = list(emotions.keys())
    
    # Normalize the confusion matrix to percentages
    confusion_matrix_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(8, 8))
    plt.imshow(confusion_matrix_normalized, cmap='Blues', interpolation='nearest')
    plt.title(f'Confusion Matrix for Fold {fold}')
    
    tick_marks = np.arange(len(emotion_labels))
    plt.xticks(tick_marks, emotion_labels, rotation=45)
    plt.yticks(tick_marks, emotion_labels)

    # Annotate each cell with the percentage value
    thresh = confusion_matrix_normalized.max() / 2.
    for i, j in np.ndindex(confusion_matrix_normalized.shape):
        plt.text(j, i, f'{confusion_matrix_normalized[i, j]:.1f}%', 
                 horizontalalignment='center',
                 color='white' if confusion_matrix_normalized[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join('./Images/', f'confusion_matrix_{fold}.png'))
    plt.clf()  # Clear the plot
 
 
def pformat_dict(d, indent=0):
    fstr = ""
    for key, value in d.items():
        fstr += '\n' + '  ' * indent + str(key) + ":"
        if isinstance(value, Mapping):
            fstr += pformat_dict(value, indent+1)
        else:
            fstr += ' ' + str(value)
    return fstr


