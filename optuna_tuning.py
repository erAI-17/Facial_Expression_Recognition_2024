import optuna
import torch
from utils.args import args
from models import efficientnet_b2, AttentionFusion1D
from tasks.emotion_recognition_task import EmotionRecognition
from utils.loaders import CalD3R_MenD3s_Dataset
from torch.utils.data import DataLoader

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)

    # Update args with suggested hyperparameters
    args.models['RGB'].lr = lr
    args.models['RGB'].weight_decay = weight_decay
    args.models['RGB'].dropout = dropout
    args.models['DEPTH'].lr = lr
    args.models['DEPTH'].weight_decay = weight_decay
    args.models['DEPTH'].dropout = dropout
    args.models['FUSION'].lr = lr
    args.models['FUSION'].weight_decay = weight_decay
    args.models['FUSION'].dropout = dropout

    # Initialize models
    rgb_model = efficientnet_b2()
    depth_model = efficientnet_b2()
    fusion_model = AttentionFusion1D(rgb_model, depth_model)

    models = {
        'RGB': rgb_model,
        'DEPTH': depth_model,
        'FUSION': fusion_model
    }

    # Initialize task
    emotion_classifier = EmotionRecognition(
        name="emotion-classifier",
        models=models,
        batch_size=args.batch_size,
        total_batch=args.total_batch,
        models_dir=args.models_dir,
        scaler=torch.cuda.amp.GradScaler(),
        class_weights=torch.FloatTensor([1.0] * 7).to('cuda'),
        model_args=args.models,
        args=args
    )

    # Load data
    train_loader = DataLoader(
        CalD3R_MenD3s_Dataset(args.dataset.name, args.modality, 'train', args.dataset, transform={}),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataset.workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        CalD3R_MenD3s_Dataset(args.dataset.name, args.modality, 'val', args.dataset, transform={}),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataset.workers,
        pin_memory=True,
        drop_last=True
    )

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emotion_classifier.load_on_gpu(device)
    emotion_classifier.train(True)
    emotion_classifier.zero_grad()

    for epoch in range(args.train.num_iter):
        for batch in train_loader:
            data, labels = batch
            data = {mod: data[mod].to(device) for mod in data}
            labels = labels.to(device)

            with torch.cuda.amp.autocast():
                logits, features = emotion_classifier.forward(data)
                emotion_classifier.compute_loss(logits, labels, features['late'])

            emotion_classifier.backward()
            emotion_classifier.step()
            emotion_classifier.zero_grad()

        # Validation
        val_metrics = validate(emotion_classifier, val_loader, device, epoch)
        trial.report(val_metrics['top1'], epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_metrics['top1']

def validate(emotion_classifier, val_loader, device, epoch):
    emotion_classifier.reset_acc()
    emotion_classifier.train(False)

    with torch.no_grad():
        for data, labels in val_loader:
            data = {mod: data[mod].to(device) for mod in data}
            labels = labels.to(device)

            with torch.cuda.amp.autocast():
                logits, _ = emotion_classifier.forward(data)

            emotion_classifier.compute_accuracy(logits, labels)

    return {'top1': emotion_classifier.accuracy.avg[1], 'top5': emotion_classifier.accuracy.avg[5]}

if __name__ == "__main__":
    # Save the study to a file
    study_name = 'example_study'  # Unique identifier for the study
    storage_name = f'sqlite:///{study_name}.db'
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=100)    

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    