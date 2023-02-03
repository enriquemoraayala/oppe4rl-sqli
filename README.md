# Introduction 
Reinforcement Learning for SQL Injection attack. This project contains the code to make experimentation about the stated problem.

# Getting Started

To develop and run code, you need to have **Miniconda** installed. In the project root run
```bash
conda env create -f environment.yaml
```
Then activate the environment with
```bash
conda activate gsoc-ds-rl-sql-injection-itinnovation
```

To run experiments in Azure Machine Learning you need to add required environment variables into an `.env` file:
```env
TENANT_ID=""
AML_WORKSPACE_NAME=""
AML_SUBSCRIPTION_ID=""
AML_RESOURCE_GROUP=""
AML_COMPUTE_NAME=""
```

And then with **conda environment activated** and **from scr/ directory** run an Azure Machine Learning Experiment submitter, for example,
```python
python -m train_dqn_agent_aml
```

The rest of document assumes the environment is activated.

# Launching experiments

We have developed three posibilities for doing experimentation:
1. Local run
2. Azure Machine Learning run
3. Ray RLLib run

The three processes run a Python script that will take the parameters from two configuration files:
* environment configuration: configuration that defines the environment that will be used.
* agent configuration: configuration related to the agent/learner.

## 1. Local run
From `src` folder, run `train_agent` pointing to the configuration files:
```ps
python -m train_agent --from-config-files --env-config "config/env.json" --agent-config "config/agent.json
```

The environment configuration `env.json` file must have the following information
```json
{
    "steps": 30, //int
    "payloads": "sqli-1waf.csv", //str
    "allowPayloadActions": false, //bool
    "featureExtractor": "SqlTermProportionFeatureExtractor", //["SqlSimpleFeatureExtractor", "SqlTermProportionFeatureExtractor"]
    "reward": "binary",  //["original", "probability", "binary"]
    "rewardWin": 10 //int
}
```

For the runs not with RLLib, i.e. local run and Azure Machine Learning run, the agent configuration has to include
```json
{
    "agent": "dqn", //["random", "dqn"]
    "episodes": 500 //int
}
```

Among the generated outputs, we will find `outputs/history.csv` with the full information of each step of every episode.

## 2. Azure Machine Learning run

Running the experiment in Azure Machine Learning does the same execution as in the local case, while the experiment and its metrics are registered in Machine Learning Workspace for efficient Experiment Tracking.

- Log into Azure CLI running az login
- Create and/or activate conda environment defined in environment.yaml
- Fill the configuration files as explained previously
- From `src` run 
    ```ps 
    python -m train_agent_aml
    ```
    and the experiment will be submitted to Azure, where will be run.

We can then check the experiment in Azure Machine Learning Workspace:
- Go to Azure Machine Learning Workspace
- Go to Jobs -> {ExperimentName} -> {RunId} -> Metrics to see the generated metrics
- Go to Jobs -> {ExperimentName} -> {RunId} -> Outputs+logs to see the logs and files generated during the run
![image](/docs/assets/azureml-experiment-run.png) ![image](/docs/assets/azureml-experiment-metrics.png)

## 3. RLLib run (Windows is in beta)

We can also make use of Ray's RLLib to run the experiments. To run, from `src` run
```ps
python -m train_agent_rllib --env-config "config/env.json" #--local-mode to allow debugging
```

In this case, the agent configuration must be included in the code itself because of the following:
- Agent selection from configuration file is not available (PPO, DQN, ...).
- Configuration of the given agent by file is allowed. However, configuration parameters for each agent is different.
- If we want to make use of Tune for hyperparameter selection, the search-space has to be defined in the code.

