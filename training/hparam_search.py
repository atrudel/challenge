from ray import tune
from ray.tune.schedulers import ASHAScheduler
from training.model_training import run



def launch_ray_tune(config, train_function, num_samples, resources):
    max_num_epochs = config['epochs']

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2
    )
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_function),
            resources=resources
        ),
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples
        ),
        param_space=config
    )
    results = tuner.fit()
    best_result = results.get_best_result("val_loss", "min")
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["val_loss"]))
    return best_result


if __name__ == '__main__':
    num_samples = 10

    config = {
        "epochs": 10,
        "batch_size": tune.choice([32, 64, 128]),
        "n_hidden": tune.choice([56, 64, 72]),
        "n_input": 86,
        "dropout": tune.uniform(0.2, 0.8),
        "learning_rate": tune.loguniform(1e-4, 1e-1)
    }
    resources = {"cpu": 10} # Add GPU
    launch_ray_tune(config, run, num_samples, resources)