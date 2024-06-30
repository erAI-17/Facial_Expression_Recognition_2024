import os
from datetime import datetime
from pathlib import Path
import torch
from abc import ABCMeta, abstractmethod
from utils.logger import logger

from typing import Dict, Optional


class Task(torch.nn.Module, metaclass=ABCMeta): 
    #? By specifying Abstract Base Classes as the metaclass, you are making Task an abstract base class. 
    #? So, you can define abstract methods within Task that must be implemented by any subclass of Task.
    #?For example, by writing :
    # @abstractmethod
    #def compute_loss(self) -> None:
    #? it means that all subclasses of Task class are FORCED to implement method "compute_loss"
    """
    Task is the abstract class which needs to be implemented for every different task present in the model
    (i.e. classification, self-supervision). 
    #*It saves a models for every modality.

    """

    #?methods with double underscores at the beginning and end of their names, such as  __getitem__ , are known as **dunder methods** (short for "double underscore methods"). 
    #? These methods are special methods that are predefined by Python to provide certain functionality and to enable the use of built-in operations with custom objects.
    #?For example,  __getitem__  method is a special method that allows instances of a class to use the indexing syntax ( obj[index] )
    #?     Other Common Dunder Methods 
    
    #? -  __init__ : Constructor method, called when an instance is created. 
    #? -  __str__ : Called by the  str()  function and by the  print  statement to get a string representation of the object. 
    #? -  __len__ : Called by the  len()  function to get the length of the object. 
    #? -  __iter__ : Returns an iterator object, enabling iteration over the object. 
    
    def __init__(
        self,
        name: str,
        task_models: Dict[str, torch.nn.Module],
        batch_size: int,
        total_batch: int,
        models_dir: str,
        args,
        **kwargs,
    ) -> None:
        
        """
        Parameters
        ----------
        name : str
            name of the task e.g. action_classifier, domain_classifier...
        task_models : Dict[str, torch.nn.module]
            torch models, one for each different modality adopted by the task. DICTIONARY : {modality (str), model (nn.module)}
        batch_size : int
            actual batch size in the forward
        total_batch : int
            batch size simulated via gradient accumulation
        models_dir : str
            directory where the models are stored when saved
        """
        
        super().__init__() #? the  super().__init__()  call is used to invoke the constructor of its parent class
                           #? in this case the constructor of the parent class doesn't take any parameter. 
        self.name = name
        self.task_models = task_models
        self.modalities = list(self.task_models.keys())
        self.batch_size = batch_size
        self.total_batch = total_batch
        self.models_dir = models_dir
        # Number of training iterations
        self.current_iter = 0
        # Index of the best validation accuracy
        self.best_iter = 0
        # Best validation accuracy
        self.best_iter_score = 0
        # Validation accuracy of the last iteration
        self.last_iter_acc = 0
        self.model_count = 1
        # Other arguments
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def compute_loss(self) -> None:
        """Compute the loss for this task"""
        pass

    def load_on_gpu(self, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """Load all the models on the GPU(s) using DataParallel.

        Parameters
        ----------
        device : torch.device, optional
            the device to move the models on, by default torch.device('cuda')
        """
        for modality, model in self.task_models.items():
            self.task_models[modality] = torch.nn.DataParallel(model).to(device)
            
    @abstractmethod
    def step(self):
        """Perform the optimization step once all the gradients of the gradient accumulation are accumulated."""
        pass

    def __restore_checkpoint(self, m: str, path: str):
        """Restore a checkpoint from a saved model path.

        Parameters
        ----------
        m : str
            modality to load from
        path : str
            path to load from
        """
        logger.info("Restoring {} for modality {} from {}".format(self.name, m, path))

        checkpoint = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Restore the state of the task
        self.current_iter = checkpoint["iteration"]
        self.best_iter = checkpoint["best_iter"]
        self.best_iter_score = checkpoint["best_iter_score"]
        self.last_iter_acc = checkpoint["acc_mean"]

        # Restore the model parameters
        self.task_models[m].load_state_dict(checkpoint["model_state_dict"], strict=True)
        # Restore the optimizer parameters
        self.optimizer[m].load_state_dict(checkpoint["optimizer_state_dict"])

        try:
            self.model_count = checkpoint["last_model_count_saved"]
            self.model_count = self.model_count + 1 if self.model_count < 9 else 1
        except KeyError:
            # for compatibility with models saved before refactoring
            self.model_count = 1

        logger.info(
            f"{m}-Model for {self.name} restored at iter {self.current_iter}\n"
            f"Best accuracy on val: {self.best_iter_score:.2f} at iter {self.best_iter}\n"
            f"Last accuracy on val: {self.last_iter_acc:.2f}\n"
            f"Last loss: {checkpoint['loss_mean']:.2f}"
        )

    def load_model(self, path: str, idx: int):
        """Load a specific model (idx-one) among the last 9 saved.

        Load a specific model (idx-one) among the last 9 saved from a specific path,
        might be overwritten in case the task requires it.

        Parameters
        ----------
        path : str
            directory to load models from
        idx : int
            index of the model to load
        """
        # List all the files in the path in chronological order (1st is most recent, last is less recent)
        last_dir = Path(
            list(
                sorted(
                    Path(path).iterdir(),
                    key=lambda date: datetime.strptime(os.path.basename(os.path.normpath(date)), "%b%d_%H-%M-%S"),
                )
            )[-1]
        )
        last_models_dir = last_dir.iterdir()

        for m in self.modalities:
            # Get the correct model (modality, name, idx)
            model = list(
                filter(
                    lambda x: m == x.name.split(".")[0].split("_")[-2]
                    and self.name == x.name.split(".")[0].split("_")[-3]
                    and str(idx) == x.name.split(".")[0].split("_")[-1],
                    last_models_dir,
                )
            )[0].name
            model_path = os.path.join(str(last_dir), model)

            self.__restore_checkpoint(model_path)

    def load_last_model(self, path: str):
        """Load the last model from a specific path.

        Parameters
        ----------
        path : str
            directory to load models from
        """
        # List all the files in the path in chronological order (1st is most recent, last is less recent)
        last_models_dir = list(
            sorted(
                Path(path).iterdir(),
                key=lambda date: datetime.strptime(os.path.basename(os.path.normpath(date)), "%b%d_%H-%M-%S"),
            )
        )[-1]
        saved_models = [x for x in reversed(sorted(Path(last_models_dir).iterdir(), key=os.path.getmtime))]

        for m in self.modalities:
            # Get the correct model (modality, name, idx)
            model = list(
                filter(
                    lambda x: m == x.name.split(".")[0].split("_")[-2]
                    and self.name == x.name.split(".")[0].split("_")[-3],
                    saved_models,
                )
            )[0].name

            model_path = os.path.join(last_models_dir, model)
            self.__restore_checkpoint(m, model_path)

    def save_model(self, current_iter: int, last_iter_acc: float, prefix: Optional[str] = None):
        """Save the model.

        Parameters
        ----------
        current_iter : int
            current iteration in which the model is going to be saved
        last_iter_acc : float
            accuracy reached in the last iteration
        prefix : Optional[str], optional
            string to be put as a prefix to filename of the model to be saved, by default None
        """
        for m in self.modalities:
            # build the filename of the model
            if prefix is not None:
                filename = prefix + "_" + self.name + "_" + m + "_" + str(self.model_count) + ".pth"
            else:
                filename = self.name + "_" + m + "_" + str(self.model_count) + ".pth"

            if not os.path.exists(os.path.join(self.models_dir, self.args.experiment_dir)):
                os.makedirs(os.path.join(self.models_dir, self.args.experiment_dir))

            try:
                torch.save(
                    {
                        "iteration": current_iter,
                        "best_iter": self.best_iter,
                        "best_iter_score": self.best_iter_score,
                        "acc_mean": last_iter_acc,
                        "loss_mean": self.loss.acc,
                        "model_state_dict": self.task_models[m].state_dict(),
                        "optimizer_state_dict": self.optimizer[m].state_dict(),
                        "last_model_count_saved": self.model_count,
                    },
                    os.path.join(self.models_dir, self.args.experiment_dir, filename),
                )
                self.model_count = self.model_count + 1 if self.model_count < 9 else 1

            except Exception as e:
                logger.error("An error occurred while saving the checkpoint: ")
                logger.error(e)

    def train(self, mode: bool = True):
        """Activate the training in all models.

        Activate the training in all models (when training, DropOut is active, BatchNorm updates itself)
        (when not training, BatchNorm is freezed, DropOut disabled).

        Parameters
        ----------
        mode : bool, optional
            train mode, by default True
        """
        for model in self.task_models.values():
            model.train(mode)

    def zero_grad(self):
        """Reset the gradient when gradient accumulation is finished."""
        for m in self.modalities:
            self.optimizer[m].zero_grad()

    def check_grad(self):
        """Check that the gradients of the model are not over a certain threshold.
        Imagine you are training a neural network, and you notice that the training process is unstable, with the loss fluctuating wildly or the model failing to converge. 
        One possible reason could be large gradients causing large updates to the model parameters.  
        By using the  check_grad  method, you can identify which parameters have large gradients and take appropriate actions,
        such as gradient clipping or adjusting the learning rate. """
        for m in self.modalities:
            for name, param in self.task_models[m].named_parameters(): #?named_parameters()  returns an iterator over the model's parameters, yielding both the name and the parameter itself
                if param.requires_grad and param.grad is not None: #?For each parameter, it checks if the parameter requires gradients ( param.requires_grad ) and if the gradient is not  None. 
                                                                   #? This ensures that only parameters that are involved in the gradient computation are checked.
                    if param.grad.norm(2).item() > 25:
                        logger.info(f"Param {name} has a gradient whose L2 norm is over 25")

    def __str__(self) -> str:
        return self.name
