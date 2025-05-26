from transformers.trainer import *
from transformers.trainer import _is_peft_model
from utils.training_utils import DecoupledLionW

class LionTrainer(Trainer):

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        optimizer_cls = DecoupledLionW
        optimizer_kwargs = {
            "lr": self.args.learning_rate,
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
            "weight_decay": self.args.weight_decay,
        }
        self.optimizer = optimizer_cls(filter(lambda p: p.requires_grad, opt_model.parameters()), **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.create_optimizer()
        if IS_SAGEMAKER_MP_POST_1_10 and smp.state.cfg.fp16:
            # If smp >= 1.10 and fp16 is enabled, we unwrap the optimizer
            optimizer = self.optimizer.optimizer
        else:
            optimizer = self.optimizer
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

class DebugLionTrainer(LionTrainer):
    pass