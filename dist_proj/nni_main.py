from nni.experiment import Experiment

search_space = {
    "learning_rate": {"_type": "choice", "_value": [1e-5, 3e-5, 5e-5, 1e-4, 2e-4]},
    "batch_size_per_gpu": {"_type": "choice", "_value": [8, 16, 32]},
    "weight_decay": {"_type": "choice", "_value": [0.05, 0.1, 0.2, 0.3]},
    "dropout_prob": {"_type": "choice", "_value": [0.1, 0.2, 0.5]},
    "focal_loss": {"_type": "choice", "_value": [True, False]},
}

if __name__ == '__main__':
    experiment = Experiment('local')
    experiment.config.experiment_name = 'dist_classify'
    experiment.config.search_space = search_space
    experiment.config.trial_command = r"""
    sh train_dist_classify.sh
    """
    experiment.config.trial_code_directory = '.'
    experiment.config.experiment_working_directory = f'./pl_log/{experiment.config.experiment_name}/nni'
    
    # tuner
    # experiment.config.tuner.name = 'GridSearch'
    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args = {
        'optimize_mode': 'maximize'
    }
    
    # trail setting
    experiment.config.max_trial_number = 30
    experiment.config.trial_concurrency = 1
    experiment.config.max_experiment_duration = '24h'
    experiment.config.nni_manager_ip = 'local'
    
    # assessor
    experiment.config.assessor.name = 'Curvefitting'
    experiment.config.assessor.class_args = {
        'epoch_num': 5,
        'start_step': 2,
        'threshold': 0.9,
        'gap': 1,
    }
    
    experiment.run(6006)