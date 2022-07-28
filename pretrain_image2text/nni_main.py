from nni.experiment import Experiment

search_space = {
    "learning_rate": {"_type": "choice", "_value": [1e-5, 3e-5, 5e-5]},
    "batch_size": {"_type": "choice", "_value": [16, 32, 64, 128]},
}

if __name__ == '__main__':
    experiment = Experiment('local')
    experiment.config.experiment_name = 'cluener_chinese-roberta-wwm-ext-large'
    experiment.config.search_space = search_space
    experiment.config.trial_command = r"""
    bash run_bert_fit.sh
    """
    experiment.config.trial_code_directory = '.'
    experiment.config.experiment_working_directory = f'./pl_log/{experiment.config.experiment_name}/nni'
    
    # tuner
    experiment.config.tuner.name = 'GridSearch'
    
    # trail setting
    experiment.config.max_trial_number = 30
    experiment.config.trial_concurrency = 1
    experiment.config.max_experiment_duration = '5h'
    experiment.config.nni_manager_ip = 'local'
    
    # assessor
    # experiment.config.assessor.name = 'Curvefitting'
    # experiment.config.assessor.class_args = {
    #     'epoch_num': 15,
    #     'start_step': 5,
    #     'threshold': 0.9,
    #     'gap': 1,
    # }
    
    experiment.run(8080)