

def launch_ray_tune(config, train_function, num_samples, resources):
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler

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
    from ray import tune
    from graph.models.rgcn import RGCN
    from training.model_training_torch_geometric import launch_experiment

    num_samples = 20
    hparams = {
        'epochs': 10,
        'batch_size': tune.choice([32, 64, 128]),
        'learning_rate': tune.loguniform(1e-4, 1e-1),
        'num_hidden_layers': tune.choice([0, 1, 3, 5]),
        'num_node_features': 86,
        'num_relations': 4,
        'num_bases': tune.choice([10, 20, 30]),
        'num_blocks': None,
        'hidden_dim': tune.choice([56, 64, 72, 84]),
        'dropout': tune.uniform(0, 0.5),
        'aggr': 'mean',
        'pooling': tune.choice(['max', 'sum', 'mean'])
    }
    resources = {"cpu": 1}

    def train_rgcn(hparams):
        model = RGCN(hparams)
        best_val_loss = launch_experiment(model, hparams, seed=None, search=True)
        return best_val_loss


    best_result = launch_ray_tune(hparams, train_rgcn, num_samples, resources)
    print(f"Best_result")
    print(best_result)