
from fedml.workflow.customized_jobs.customized_base_job import CustomizedBaseJob


class TrainJob(CustomizedBaseJob):
    def __init__(self, name, job_yaml_absolute_path=None, job_api_key=None, config_version=None,
                 local_on_prem_host="localhost", local_on_prem_port=80):
        super().__init__(name, job_yaml_absolute_path=job_yaml_absolute_path,
                         job_api_key=job_api_key, config_version=config_version,
                         local_on_prem_host=local_on_prem_host, local_on_prem_port=local_on_prem_port)

    def run(self):
        super().run()

    def status(self):
        return super().status()

    def kill(self):
        super().kill()

