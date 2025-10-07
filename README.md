# NYC and JFK Airport Taxi Data Analysis
Exploring and visualising New York City Taxi trip data (2011-2025). With modelling of daily and hourly taxi counts for JFK Airport using linear regresssion, xgboost and linear regression boosted on the residuals.  

## Objective

Our objective was to do some basic exploration of NYC taxi data using choropleths. Then to explore fitting various models two time series for the taxi count at JFK Airport, one daily and one hourly.

## Data Sources

Data source: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), 
via [NYC Open Data](https://opendata.cityofnewyork.us/).  
© City of New York. Data made available under the NYC Open Data Terms of Use.

## Usage

Clone the repository:
```bash
git clone https://github.com/William-Ogilvie/NYC-taxi-data-analysis.git
cd NYC-taxi-data-analysis
```



- The data you will need are the Taxi Zone Shapefile (PARQUET), Taxi Zone Lookup Table (CSV) and the Yellow Taxi Trip Records (PARQUET) for Januaray 2025. (change once add more data)
- All of which can be downloaded from [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
- Extract any zip files and place all data in a data/raw directory.

To run the notebook you will need to install the dependencies in requirements.txt. Create a virtual enviroment and install all dependencies:

```bash
pyton -m venv venv
source venv/bin/activate   # on MAC/Linux
venv/Scripts/activate      # On Windows
```
```bash
pip install -r requirements.txt
```

All the scripts in src/ are under a package called jfk_taxis (the setup is in setup.py). This will need to be installed on top. The -e means the pacakge is in editable mode so any changes will be made available without a reinstall.  

```bash
pip install -e .
```

You will now be able to run the Notebooks. Note any output will be stored in a reports directory. 

```bash
jupyter lab
```

NOte setup.py will now test for gpu/cpu and set xgboost to appropriate device

## Reults / Key Findings

We initally perform EDA for the entire of the NYC taxi data, primarily using choropleths. We then shift our focus to JFK Airport and create two time series that count the total number of taxis at the Airport, one daily, one hourly. We explore these time series to try and find relevant predictive features that we could use for models. Our intial exploration suggests that daily and weekly fourier features could be good for our hourly time series, and weekly and yearly fourier features for our daily time series. We also get back a considerable number of lags for both time series that cover at least one year into the past. 

Then we fit some inital models to these featuers. We test three strands of model linear regression, xgboost (usually reffered to as our non linear model in notebooks) and boosted linear regression on the residuals (so xgboost is fitted to the residuals of a linear regression, and added to its predictions). We will use a Naive baseline of just predicting the current value as the same as that one time step (day or hour) ago. The use of this baseline originally was from trial and error among others like 1 week ago. However it is supported via the SHAP values as most models have a very large mean SHAP value for lag_1. Suggesting that the previous time step has considerable predictive power. (comment on findings here).

Later we explore feature importance using SHAP values for all the models. We take the top 30 features by average SHAP value and see if they retain most of the predictive signal. It turns out for xgboost the reduced feature set is similarly effective. However for just pure linear regression it does seem to be considerably worse and potentially not worth using as the computational savings are smaller compared to with xgboost. 

Finally we perform bayesian hyperparameter tuning with optuna. I personally ran this in on AWS EC2 instance over a few days to do 1000 trials (maybe change this idk?) but the improvements seem to be minimal unfortunately (maybe?!).

We conclude with the best models for the two cases (add here). 



Note explain the install of XGBoost for gpus carefully because we assume that u run one

Comment on fact we expected fourier features to be important, turns out according to SHAP values actaully lags matter way way more.

Make sure to explain that in terms of hourly lags we can't use all of them so we take first 300 approx and throw in roughly half year and full year ones, with full year being significant.

Explain time zone conversions maybe? like converting to UTC to avoid DST errors

maybe comment on how offsets work? every 30 days with a bit of jitter, so in total 7 offsets


## Testing guide

before running tests you may need to run setup.py to ensure config is configured correctly for your PC

## EC2 instance guide

As the hyperparameter tuning can be computationally expensive I decided to have it run on an EC2 instance in AWS rather than on my host machine. To do this I setup a S3 bucket to store the time series csvs as well as the outputs of the hyperparameter tuning. Then I create an EC2 instance with permissions to access the S3 bucket. I use the Deep Learning Ubuntu 20.04 AMI as it has already has the Nvidia drivers installed (if you use an instance with a GPU) as well as docker, git and AWS CLI. It will need to have an IAM role with the AmazonS3FullAccesss policy when created. 

Once connected to the instance clone the repository from git, (git clone https://github.com/William-Ogilvie/NYC-taxi-data-analysis.git) then use docker to build the image and run the container (add commands here). You will then want to run the setup.py script to create all necessary directories and ensure that xgboost is set to either cpu or cuda as appropriate. Then close the container. Inside the S3 bucket place the full time series data (ts_daily2011-2025.csv, ts_hourly2011-2025.csv) inside a data/time_series directory. To download this directory to our EC2 instance first do ```bash aws s3 ls ``` to see all s3 buckets. Then ```bash aws s3 sync://BUCKET_NAME/data/time_series data/processed ```. 

upload:
mkdir -p ~/outputs
# (…produce some files…)
aws s3 sync ~/outputs s3://YOUR-BUCKET/outputs/test-run --only-show-errors

1) Make an S3 bucket (for datasets / results)

Console → S3 → Create bucket

Name: e.g. jfk-taxi-data-<yourname>

Region: eu-west-2 (London) (keep things in one region)

Block public access: on

Create bucket

You can upload files in the UI later (or upload from the EC2 instance).

2) Launch an EC2 instance (GPU or CPU)

Console → EC2 → Launch instance

Name: jfk-taxi-runner

AMI:

For GPU (easiest): Deep Learning AMI (Ubuntu 22.04) – already has NVIDIA drivers + Docker.

For CPU-only: Ubuntu Server 22.04 LTS is fine.

Instance type:

GPU: g4dn.xlarge (T4) or g5.xlarge (A10G)

CPU: c7i.large (for example)

Key pair: Create a new one (download the .pem).

Network settings / Security group:

Inbound: SSH (22) from My IP

(Optional) For Jupyter: add Custom TCP 8888 from My IP

Storage: bump disk (e.g. 100–200 GB gp3) if you’ll download data locally.

IAM role (click “Create new IAM role” in the wizard’s dropdown):

Trusted entity: EC2

Attach policies:

AmazonS3FullAccess (or a tighter custom policy to your bucket)

(Optional) AmazonEC2ContainerRegistryReadOnly if you’ll pull from ECR

Create role and select it in the EC2 wizard.

User data: leave empty for now (you can add later to automate).

Click Launch instance.

3) Connect to the box

EC2 → Instances → select your instance → Connect

If EC2 Instance Connect is available, use it (browser shell, no keys).

Otherwise: use your .pem with SSH:

ssh -i path/to/your-key.pem ubuntu@<EC2_PUBLIC_IP>


From here on, the few commands you run are inside the EC2 shell.

4) Install Docker (only if you picked plain Ubuntu)

If you used the Deep Learning AMI, skip to step 5.

On Ubuntu:

sudo apt-get update
sudo apt-get install -y docker.io git
sudo usermod -aG docker $USER
newgrp docker


For GPU on plain Ubuntu, add the NVIDIA Container Toolkit:

distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
| sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
| sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
| sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

5) Get your code onto the instance

Simplest: git clone your repo:

git clone <YOUR_REPO_URL> project
cd project


(If private, use an SSH key or a GitHub token.)

If your data’s in S3 and you attached an IAM role, you can pull it from the instance using the S3 console (browser upload/download) or quickly install AWS CLI on the instance (sudo apt-get install -y awscli) and run:

aws s3 sync s3://your-bucket/data ./data

6) Build your Docker image on EC2

From the project folder that contains your Dockerfile:

# GPU build (uses your environment_gpu.yml)
docker build -t jfk-taxi:gpu --build-arg ENV_FILE=environment_gpu.yml .

# or CPU:
# docker build -t jfk-taxi:cpu --build-arg ENV_FILE=environment_cpu.yml .

7) Run your container

Mount your repo so code changes persist (standard dev flow):

GPU:

docker run --rm -it --gpus all \
  -v "$PWD:/app" -w /app \
  jfk-taxi:gpu \
  python scripts/train.py --data /app/data --out /app/outputs


CPU:

docker run --rm -it \
  -v "$PWD:/app" -w /app \
  jfk-taxi:cpu \
  python scripts/train.py --data /app/data --out /app/outputs


For JupyterLab on the instance:

docker run --rm -it --gpus all -p 8888:8888 \
  -v "$PWD:/app" -w /app \
  jfk-taxi:gpu \
  jupyter lab --ip=0.0.0.0 --no-browser


Open http://<EC2_PUBLIC_IP>:8888 in your browser (you allowed port 8888 to your IP).

8) Save results to S3 (UI or from the instance)

In the S3 console, upload files from your outputs/ folder.

Or, from the instance (if CLI installed and role attached):

aws s3 sync ./outputs s3://your-bucket/outputs/exp1

9) (Optional) Automate overnight

In the EC2 launch wizard, you can paste a script into User data so the instance:

pulls data from S3,

runs your Docker command,

uploads results back to S3,

then shuts down.

This avoids babysitting the run.

10) Clean up

When you’re done, in the EC2 console:

Stop the instance (to pause costs) or Terminate (to delete).

Delete the security group, key pair, and (optional) S3 bucket if you created them just for tests.

Which button where?

If you ever get lost:

EC2 → Instances (start/stop/connect)

EC2 → Security Groups (open/close ports)

IAM → Roles (attach S3/ECR perms to your instance)

S3 → Buckets (upload/download datasets & results)

(Optional) ECR → Repositories (if you want to store your image in AWS instead of building on the box)

You absolutely don’t need the AWS CLI on your laptop for this flow. The only commands you’ll type are inside the EC2 shell (Linux commands to git clone, docker build, docker run).