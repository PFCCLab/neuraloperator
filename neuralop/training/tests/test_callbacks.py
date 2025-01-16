import shutil
from pathlib import Path

import paddle
from paddle import nn
from paddle.io import Dataset, DataLoader

from neuralop import Trainer, LpLoss, H1Loss, CheckpointCallback
from neuralop.models.base_model import BaseModel


class DummyDataset(Dataset):
    # Simple linear regression problem, PyTorch style

    def __init__(self, n_examples):
        super().__init__()

        self.X = paddle.randn((n_examples, 50))
        self.y = paddle.randn((n_examples, 1))

    def __getitem__(self, idx):
        return {"x": self.X[idx], "y": self.y[idx]}

    def __len__(self):
        return self.X.shape[0]


class DummyModel(BaseModel, name="Dummy"):
    """
    Simple linear model to mock-up our model API
    """

    def __init__(self, features, **kwargs):
        super().__init__()
        self.net = nn.Linear(features, 1)

    def forward(self, x, **kwargs):
        """
        Throw out extra args as in FNO and other models
        """
        return self.net(x)


def test_model_checkpoint_saves():
    save_pth = Path("./test_checkpoints")

    model = DummyModel(50)

    train_loader = DataLoader(DummyDataset(100))

    trainer = Trainer(
        model=model,
        n_epochs=5,
        callbacks=[
            CheckpointCallback(
                save_dir=save_pth, save_optimizer=True, save_scheduler=True
            )
        ],
    )

    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=8e-3, T_max=30)
    optimizer = paddle.optimizer.Adam(
        learning_rate=scheduler, parameters=model.parameters(), weight_decay=1e-4
    )

    # Creating the losses
    l2loss = LpLoss(d=2, p=2)

    trainer.train(
        train_loader=train_loader,
        test_loaders={},
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=None,
        training_loss=l2loss,
        eval_losses=None,
    )

    for file_ext in [
        "model_state_dict.pdmodel",
        "model_metadata.pkl",
        "optimizer.pdopt",
        "scheduler.pdopt",
    ]:
        file_pth = save_pth / file_ext
        assert file_pth.exists()

    # clean up dummy checkpoint directory after testing
    shutil.rmtree("./test_checkpoints")


def test_model_checkpoint_and_resume():
    save_pth = Path("./full_states")
    model = DummyModel(50)

    train_loader = DataLoader(DummyDataset(100))
    test_loader = DataLoader(DummyDataset(20))

    trainer = Trainer(
        model=model,
        n_epochs=5,
        callbacks=[
            CheckpointCallback(
                save_dir=save_pth,
                save_optimizer=True,
                save_scheduler=True,
                save_best="h1",
            )  # monitor h1 loss
        ],
    )

    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=8e-3, T_max=30)
    optimizer = paddle.optimizer.Adam(
        learning_rate=scheduler, parameters=model.parameters(), weight_decay=1e-4
    )

    # Creating the losses
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)

    eval_losses = {"h1": h1loss, "l2": l2loss}

    trainer.train(
        train_loader=train_loader,
        test_loaders={"": test_loader},
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=None,
        training_loss=l2loss,
        eval_losses=eval_losses,
    )

    for file_ext in [
        "best_model_state_dict.pdmodel",
        "best_model_metadata.pkl",
        "optimizer.pdopt",
        "scheduler.pdopt",
    ]:
        file_pth = save_pth / file_ext
        assert file_pth.exists()

    # Resume from checkpoint
    trainer = Trainer(
        model=model,
        n_epochs=5,
        callbacks=[
            CheckpointCallback(
                save_dir="./checkpoints", resume_from_dir="./full_states"
            )
        ],
    )

    trainer.train(
        train_loader=train_loader,
        test_loaders={"": test_loader},
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=None,
        training_loss=l2loss,
        eval_losses=eval_losses,
    )

    # clean up dummy checkpoint directory after testing
    shutil.rmtree(save_pth)


# ensure that model accuracy after loading from checkpoint
# is comparable to accuracy at time of save
def test_load_from_checkpoint():
    model = DummyModel(50)

    train_loader = DataLoader(DummyDataset(100))
    test_loader = DataLoader(DummyDataset(20))

    trainer = Trainer(
        model=model,
        n_epochs=5,
        callbacks=[
            CheckpointCallback(
                save_dir="./full_states",
                save_optimizer=True,
                save_scheduler=True,
                save_best="h1",
            )  # monitor h1 loss
        ],
    )

    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=8e-3, T_max=30)
    optimizer = paddle.optimizer.Adam(
        learning_rate=scheduler, parameters=model.parameters(), weight_decay=1e-4
    )

    # Creating the losses
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)

    eval_losses = {"h1": h1loss, "l2": l2loss}

    orig_model_eval_errors = trainer.train(
        train_loader=train_loader,
        test_loaders={"": test_loader},
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=None,
        training_loss=l2loss,
        eval_losses=eval_losses,
    )

    # create a new model from saved checkpoint and evaluate
    loaded_model = DummyModel.from_checkpoint(
        save_folder="./full_states", save_name="best_model"
    )
    trainer = Trainer(
        model=loaded_model,
        n_epochs=1,
    )

    loaded_model_eval_errors = trainer.evaluate(
        loss_dict=eval_losses, data_loader=test_loader
    )

    # log prefix is empty except for default underscore
    assert orig_model_eval_errors["_h1"] - loaded_model_eval_errors["_h1"] < 0.1

    # clean up dummy checkpoint directory after testing
    shutil.rmtree("./full_states")
