import os
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.authentication import InteractiveLoginAuthentication
from dotenv import load_dotenv


EXPERIMENT_NAME = "dev"

"""
ARGUMENTS
"""
env_config = "config/env.json"
agent_config = "config/agent.json"
#agent = "dqn"
#episodes = 1e+3
#steps = 30
#payloads = "sqli-1waf.csv"
#start_with_payload = "random"
#allow_payload_actions = "true"
#feature_extractor = "SqlTermProportionFeatureExtractor"
#reward = "binary"
#reward_win = 50
#strategy = "all" # "first", "random", "all"
#seed = None


def main() -> None:
    env_fpath = ".env"
    try:
        load_dotenv(env_fpath)
    except:
        print(f"Environment variable file {env_fpath} not found.")
    aml_auth = InteractiveLoginAuthentication(tenant_id=os.environ["TENANT_ID"])
    ws = Workspace.get(name=os.environ["AML_WORKSPACE_NAME"], subscription_id=os.environ["AML_SUBSCRIPTION_ID"],
                    resource_group=os.environ["AML_RESOURCE_GROUP"], auth=aml_auth)
    experiment = Experiment(workspace=ws, name=EXPERIMENT_NAME)
    config = ScriptRunConfig(source_directory='./',
                             script='train_agent.py',
                             arguments=[
                                "--from-config-files",
                                "--env-config", env_config,
                                "--agent-config", agent_config,
                                #"--agent", agent,
                                #"--episodes", episodes,
                                #"--steps", steps,
                                #"--payloads", payloads,
                                #"--start-with-payload", start_with_payload,
                                #"--allow-payload-actions", allow_payload_actions,
                                #"--feature-extractor", feature_extractor,
                                #"--reward", reward,
                                #"--reward-win", reward_win,
                                #"--strategy", strategy,
                                #"--seed", seed
                             ],
                             compute_target=os.environ["AML_COMPUTE_NAME"])

    # set up environment
    env = Environment.from_conda_specification(
        name='rl-env',
        file_path='assets/rl-env.yaml'
    )
    config.run_config.environment = env

    run = experiment.submit(config)

    aml_url = run.get_portal_url()
    print(aml_url)


if __name__ == "__main__":
    main()