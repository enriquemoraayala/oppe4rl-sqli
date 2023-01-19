import os
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.authentication import InteractiveLoginAuthentication
from dotenv import load_dotenv


if __name__ == "__main__":
    env_fpath = ".env"
    try:
        load_dotenv(env_fpath)
    except:
        print(f"Environment variable file {env_fpath} not found.")
    aml_auth = InteractiveLoginAuthentication(tenant_id=os.environ["TENANT_ID"])
    ws = Workspace.get(name=os.environ["AML_WORKSPACE_NAME"], subscription_id=os.environ["AML_SUBSCRIPTION_ID"],
                    resource_group=os.environ["AML_RESOURCE_GROUP"], auth=aml_auth)
    experiment = Experiment(workspace=ws, name='test-experiment')
    config = ScriptRunConfig(source_directory='./',
                             script='train_dqn_agent.py',
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