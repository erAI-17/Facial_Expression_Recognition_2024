import matplotlib.pyplot as plt

def plot_mean_verb_loss(file_path):
    iterations = []
    mean_verb_loss_values = []

    with open(file_path, 'r') as file:
        for line in file:
            if "Mean verb loss" in line:
               values = line.split()
               iteration = int(values[4].split('/')[0].replace("[", ""))
               mean_verb_loss = float(values[12])
               iterations.append(iteration)
               mean_verb_loss_values.append(mean_verb_loss)

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, mean_verb_loss_values, color='b', linewidth=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Mean Train Loss')
    plt.title('')
    plt.grid(True)
    
    
    # Zoom in on the plot
    plt.xlim(iterations[0], iterations[-1])  # Adjust the x-axis limits
    plt.ylim(min(mean_verb_loss_values), 1)  # Adjust the y-axis limits
    
    plt.savefig('Experiment_logs/feat_fusion_train_loss.png')
    plt.show()
    
    
def plot_val_acc(file_path):
    iterations = []
    val_accs = []

    with open(file_path, 'r') as file:
        for line in file:
            values = line.split()
            iteration = int(values[0].split('/')[0].replace("[", ""))
            val_acc = float(values[2].split('%')[0])
            iterations.append(iteration)
            val_accs.append(val_acc)

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, val_accs, color='b', linewidth=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Val Accuracy')
    plt.title('')
    plt.grid(True)
    
    
    # Zoom in on the plot
    plt.xlim(iterations[0], iterations[-1])  # Adjust the x-axis limits
    plt.ylim(min(val_accs), max(val_accs))  # Adjust the y-axis limits
    
    plt.savefig('Experiment_logs/feat_fusion_val_acc.png')
    plt.show()    

# Specify the path to your text file here
file_path1 = 'Experiment_logs/feat_fusion_train_loss.txt'
plot_mean_verb_loss(file_path1)

file_path2 = 'Experiment_logs/feat_fusion_val_acc.txt'
plot_val_acc(file_path2)

