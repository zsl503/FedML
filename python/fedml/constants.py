# FedML TRAINING PLATFORM
FEDML_TRAINING_PLATFORM_SIMULATION = "simulation"
FEDML_TRAINING_PLATFORM_CROSS_SILO = "cross_silo"
FEDML_TRAINING_PLATFORM_CROSS_DEVICE = "cross_device"
FEDML_TRAINING_PLATFORM_DISTRIBUTED = "distributed"
FEDML_TRAINING_PLATFORM_CROSS_CLOUD = "cross_cloud"
FEDML_TRAINING_PLATFORM_SERVING = "fedml_serving"

# FedML TRAINING PLATFORM TYPE
FEDML_TRAINING_PLATFORM_CROSS_SILO_TYPE = 1
FEDML_TRAINING_PLATFORM_SIMULATION_TYPE = 2
FEDML_TRAINING_PLATFORM_DISTRIBUTED_TYPE = 3
FEDML_TRAINING_PLATFORM_CROSS_DEVICE_TYPE = 4
FEDML_TRAINING_PLATFORM__CROSS_CLOUD_TYPE = 5
FEDML_TRAINING_PLATFORM_SERVING_TYPE = 6

# FedML CROSS-CLOUD SCENARIO
FEDML_CROSS_CLOUD_SCENARIO_HORIZONTAL = "horizontal"
FEDML_CROSS_CLOUD_SCENARIO_HIERARCHICAL = "hierarchical"
FEDML_CROSS_CLOUD_CUSTOMIZED_HIERARCHICAL_KEY = "use_customized_hierarchical"

# FedML CROSS-SILO SCENARIO
FEDML_CROSS_SILO_SCENARIO_HORIZONTAL = "horizontal"
FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL = "hierarchical"
FEDML_CROSS_SILO_CUSTOMIZED_HIERARCHICAL_KEY = "use_customized_hierarchical"

# FedML SIMULATION TYPE
FEDML_SIMULATION_TYPE_SP = "sp"
FEDML_SIMULATION_TYPE_MPI = "MPI"
FEDML_SIMULATION_TYPE_NCCL = "NCCL"

# FedML data
FEDML_DATA_CACHE_FOLDER = "fedml_data"
FEDML_DATA_MNIST_URL = "https://fedcv.s3.us-west-1.amazonaws.com/MNIST.zip"


# FedML Algorithm
FedML_FEDERATED_OPTIMIZER_BASE_FRAMEWORK = "base_framework"
FedML_FEDERATED_OPTIMIZER_FEDAVG = "FedAvg"
FedML_FEDERATED_OPTIMIZER_FEDOPT = "FedOpt"
FedML_FEDERATED_OPTIMIZER_FEDPROX = "FedProx"
FedML_FEDERATED_OPTIMIZER_CLASSICAL_VFL = "classical_vertical"
FedML_FEDERATED_OPTIMIZER_SPLIT_NN = "split_nn"
FedML_FEDERATED_OPTIMIZER_DECENTRALIZED_FL = "decentralized_fl"
FedML_FEDERATED_OPTIMIZER_FEDGAN = "FedGAN"
FedML_FEDERATED_OPTIMIZER_FEDAVG_ROBUST = "FedAvg_robust"
FedML_FEDERATED_OPTIMIZER_FEDAVG_SEQ = "FedAvg_seq"
FedML_FEDERATED_OPTIMIZER_FEDOPT_SEQ = "FedOpt_seq"
FedML_FEDERATED_OPTIMIZER_FEDGKT = "FedGKT"
FedML_FEDERATED_OPTIMIZER_FEDNAS = "FedNAS"
FedML_FEDERATED_OPTIMIZER_FEDSEG = "FedSeg"
FedML_FEDERATED_OPTIMIZER_TURBO_AGGREGATE = "turbo_aggregate"
FedML_FEDERATED_OPTIMIZER_FEDNOVA = "FedNova"
FedML_FEDERATED_OPTIMIZER_FEDDYN = "FedDyn"
FedML_FEDERATED_OPTIMIZER_SCAFFOLD = "SCAFFOLD"
FedML_FEDERATED_OPTIMIZER_MIME = "Mime"
FedML_FEDERATED_OPTIMIZER_HIERACHICAL_FL = "HierarchicalFL"
FedML_FEDERATED_OPTIMIZER_FEDSGD = "FedSGD"
FedML_FEDERATED_OPTIMIZER_FEDLOCALSGD = "FedLocalSGD"
FedML_FEDERATED_OPTIMIZER_ASYNC_FEDAVG = "Async_FedAvg"

# FedML backend service entrypoints

# These variables define the domain and URL for the backend service in different environments
# (development, testing, and release). The domain is the IP address and port number of the backend
# service, while the URL is the complete URL including the protocol (http or https). These variables
# are used to configure the backend service connection in the code.

FEDML_BACKEND_SERVICE_URL_LOCAL = "http://34.83.130.103:18080"

FEDML_BACKEND_SERVICE_URL_DEV = "https://open-dev.fedml.ai"

FEDML_BACKEND_SERVICE_URL_TEST = "https://open-test.fedml.ai"

FEDML_BACKEND_SERVICE_URL_RELEASE = "https://open.fedml.ai"

FEDML_MQTT_DOMAIN_LOCAL = "34.83.130.103"
FEDML_MQTT_DOMAIN_DEV = "mqtt-dev.fedml.ai"
FEDML_MQTT_DOMAIN_TEST = "mqtt.fedml.ai"
FEDML_MQTT_DOMAIN_RELEASE = "mqtt.fedml.ai"

FEDML_S3_DOMAIN_LOCAL = "http://127.0.0.1:9000"
