class Logger:
    def __init__(self, **config):
        self.config = config
        self.experiment = self.config["experiment"]

        self._log_hyperparams()

    def _log_hyperparams(self):
        self.experiment.log_parameters(self.config)
