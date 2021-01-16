# Operationalizing Machine Learning

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we use AutoML to train a best model, then operationalize it by following the workflow below. Both the `Azure ML Studio` and `Python SDK` were used in this project.

This diagram provides a visual summary of the workflow:
![workflow](assets/Project_tasks.png)

**Image credit: Udacity MLEMA Nanodegree**

Here is a summary of the workflow steps. A detailed account of the step executions is in the _**Architectural Diagram**_ section.

**1. Authentication**

This step used the `az cli` interface to log in to the `AML Studio`, then create a Service Principal (SP) to access the project workspace. As Udacity provisioned AML lab environment does not have sufficient privilege to create the SP, this step was not performed.

**2. Auto ML model**

This step used AML AutoML to train a collection of classification models on this [Bank Marketing dataset](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) and present the trained models in descending order of **AUC weighted accuracy**.

**3. Deploy the best model**

In this step, the top performing model, i.e. the one with the **best** *AUC weighted accuracy* was selected for deployment, and an endpoint to interact with the model was generated.

**4. Enable logging**

This step used `az cli` interface to enable Application Insights and retrieve logs of the operational health of the deployed model endpoint.

**5. Consume model endpoints**

In this step, a provided script was run in the `az cli` interface to make a request to the deployed model endpoint and display the response received. The payload data used for testing the endpoint was also saved to a json file named `data.json` for use in conducting a benchmarking test on the endpoint.

**6. Create and publish a pipeline**

This involved creating and publishig an endpoint for the AutoML training pipeline, allowing the training process to be automated.

**7. Documentation**

In this final step, a screencast was created to show the entire process of the working ML application, along with a README.md file to describe the project and document the main steps.

## Architectural Diagram
![Architectural Diagram](assets/MLOPSArch.png)

A detailed account of the workflow steps illustrated in the architectural diagram is discussed here. The steps are grouped into 3 sub sections _**AutoML Model Training**_, _**Model Deployment**_ and _**Training Pipeline Automation**_.

### AutoML Model Training
This process consists of several steps involving setting up the training dataset and AutoML config, creating a pipeline to run the training process. The steps can be performed in either `AML Studio` or using `Python SDK`. I opted to use the project provided notebook `aml-pipelines-with-automated-machine-learning-step.ipynb` to complete the steps. This notebook is included in the project submission package. Refer to the notebook for code and step execution details. Below is an abstract of the key steps

**1. Dataset**

This code snippet below shows how the [Bank marketing dataset](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) was uploaded. It was of tabular dataset type and included the target column 'y'.

```python
found = False
key = "BankMarketing Dataset"
description_text = "Bank Marketing DataSet for Udacity Course 2"

if key in ws.datasets.keys():
        found = True
        dataset = ws.datasets[key]

if not found:
        # Create AML Dataset and register it into Workspace
        example_data = 'https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv'
        dataset = Dataset.Tabular.from_delimited_files(example_data)
        #Register Dataset in Workspace
        dataset = dataset.register(workspace=ws,
                                   name=key,
                                   description=description_text)


df = dataset.to_pandas_dataframe()
df.describe()
```

This image shows the dataset was uploaded and registered successfully for use by the AutoML training.

![Dataset](assets/dataset.png)

**2. AutoML Config**

This code snippet shows the setup of AutoML config:

```python
automl_settings = {
    "experiment_timeout_minutes": 20,
    "max_concurrent_iterations": 5,
    "primary_metric" : 'AUC_weighted'
}
automl_config = AutoMLConfig(compute_target=compute_target,
                             task = "classification",
                             training_data=dataset,
                             label_column_name="y",
                             path = project_folder,
                             enable_early_stopping= True,
                             featurization= 'auto',
                             debug_log = "automl_errors.log",
                             **automl_settings
                            )
```

and this, the setup of AutoMLStep:

```python
automl_step = AutoMLStep(
    name='automl_module',
    automl_config=automl_config,
    outputs=[metrics_data, model_data],
    allow_reuse=True)
```

**3. Training Pipeline Creation and Run**

This code snippet illustrates creation of the training pipeline:

```python
from azureml.pipeline.core import Pipeline
pipeline = Pipeline(
    description="pipeline_with_automlstep",
    workspace=ws,
    steps=[automl_step])
```

and submission of the pipeline run:

```python
pipeline_run = experiment.submit(pipeline)
```

**4. Pipeline Run Monitoring**

The pipeline run was monitored from within the notebook with the  RunDetails [Jupyter widget](http://jupyter.org/widgets) as shown here:

![Widget](assets/RunWidget.png)

The piepline run was also visible on the _**Pipelines**_ console in the `AML Studio`:

![PipelineRun](assets/piperun.png)

**5. Pipeline Run Completion**

The pipeline run successfully completed as shown in the series of screenshots presented here.

This screenshot shows the pipeline run successfully concluded after 39 minutes and 30 seconds:

![PipelineRunDone](assets/piperundone.png)

The was also displayed inside the notebook like so:

![WidgetDone](assets/RunWidgetdone.png)

On the _**Experiments**_ console in the `AML Studio`, the associated experiment `ml-experiment-1` had a Completed status with a green checkmark next to it, as shown here:

![ExpDone](assets/exprundone.png)

Clicked on the `Run 1` link to drill down to the run Graph and Pipeline run overview:

![ExpDtlDone](assets/exprundtldone.png)

The run produced a list of top performing models in descending order of _**AUC weighted accuracy**_. The best model topping the list is the one named `VotingEnsemble`:

![BestModelList](assets/bestmodellist.png)

Clicking the `VotingEnsemble` hyperlink revealed details of the best model:

![BestModeldtl](assets/bestmodeldtl1.png)

The best model details was also viewable by drilling down to `Run 1` of the `ml-experiment-1` experiment from the _**Experiments**_ console in the `AML Studio`:

![ExpDtlDone2](assets/exprundtldone2.png)

**6. Conclusion**

The _**AutoML Model Training**_ process was successfully executed with a best model ready for deployment.

### Model Deployment
The process consists of a series of steps executed in both `AML Studio` and `az cli` interface.

**1. Deploy the Best Model**

This was done by clicking the _**Deploy**_ button from the best model _**Details**_ page on the _**Experiments**_ console in the `AML Studio`. The model was deployed to an **ACI** (Azure Container Instance) with authentication enabled as shown here:

![BMDeploySub](assets/bmdepolysub1.png)

The deployment was submitted successfully:

![BMDeploystatus](assets/bmdepolysub2.png)

When the deployment was done, an endpoint was generated along with a Swagger URI. This was visible from the _**Endpoints**_ console in the `AML Studio`. Notice that `Application Insights` was not yet enabled at that point.

![BMDeploy](assets/bmdeploy.png)

**2. Enable Logging**

After the best model was successfully deployed, the next step was to run a provided python script named `logs.py` in the `az cli` interface to enable `Application Insights` and retrieve logs. This is the code snippet:

```python
from azureml.core import Workspace
from azureml.core.webservice import Webservice

# Requires the config to be downloaded first to the current working directory
ws = Workspace.from_config()

# Set with the deployment name
name = "best-model-deploy"

# load existing web service
service = Webservice(name=name, workspace=ws)

service.update(enable_app_insights=True)

logs = service.get_logs()

for line in logs.split('\n'):
    print(line)
```

For this script to execute successfully, the AML workspace configuration file `config.json` was downloaded and placed in the same folder as this script. The configuration file looks like this:

```
{
    "subscription_id": "7a5e5192-86c5-4374-9780-5ddf26e7d9e1",
    "resource_group": "aml-quickstarts-124895",
    "workspace_name": "quick-starts-ws-124895"
}
```

The screenshot below shows successful execution of the `logs.py` script with `Application Insights` enabled and logs retrieved:

![LogEnable](assets/logenable.png)

On the _**Endpoints**_ console in the AML Studio, the _**Details**_ tab of the best model page showed `Application Insights` was enabled successful with an url provided:

![LogEnabled](assets/logenabled.png)

The _**Deployment**_ tab on the same page displayed logs received:

![LogEnabled2](assets/logenabled2.png)

**3. Swagger Documentation**

Azure provides a [Swagger JSON file](https://swagger.io/) for deployed models. The Swagger URI (see the deployed model's _**Details**_ tab on the _**Endpoints**_ console in the `AML Studio`) was used to download the `swagger.json` file and saved to a folder where the scripts (`swagger.sh` and `serve.py`) for downloading Swagger Container and starting a Python web server on the local host reside.

This screenshot shows `swagger.json` was downloaded successfully from the Swagger URI and saved to the Swagger script folder:

![SwaggerDnld](assets/swagerJsonDnld.png)

Next `swagger.sh` and `serve.py` were started on the local host
to stage a swagger instance with the documentation for HTTP API of the deployed model on the local host, as shown here:

![Swagger](assets/swagger.png)

The `POST/score` method of the HTTP API looks like this:

![SwaggerPOST](assets/swaggerpost.png)

**4. Consume Model Endpoint**

Next up, the best model endpoint was put to test by executing a python script `endpoint.py` in the `az cli` interface. The script contains the model endpoint and authorization key (listed on the model's _**Consume**_ tab on the _**Endpoints**_ console in the `AML Studio`), posted a payload to the endpoint and displayed the response for the `POST` request. It also saved the payload to a json file `data.json` which was used subsequently to conduct a benchmarking test on the endpoint. The is the endpoint testing script:

```python
import requests
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = 'http://d8131649-149d-4a94-88d1-000024f32a51.southcentralus.azurecontainer.io/score'
# If the service is authenticated, set the key or token
key = 'gJaAltgcXjbyTpEDNsFd5Q1QFMXQMaQF'

# Two sets of data to score, so we get two results back
data = {"data":
        [
          {
            "age": 17,
            "campaign": 1,
            "cons.conf.idx": -46.2,
            "cons.price.idx": 92.893,
            "contact": "cellular",
            "day_of_week": "mon",
            "default": "no",
            "duration": 971,
            "education": "university.degree",
            "emp.var.rate": -1.8,
            "euribor3m": 1.299,
            "housing": "yes",
            "job": "blue-collar",
            "loan": "yes",
            "marital": "married",
            "month": "may",
            "nr.employed": 5099.1,
            "pdays": 999,
            "poutcome": "failure",
            "previous": 1
          },
          {
            "age": 87,
            "campaign": 1,
            "cons.conf.idx": -46.2,
            "cons.price.idx": 92.893,
            "contact": "cellular",
            "day_of_week": "mon",
            "default": "no",
            "duration": 471,
            "education": "university.degree",
            "emp.var.rate": -1.8,
            "euribor3m": 1.299,
            "housing": "yes",
            "job": "blue-collar",
            "loan": "yes",
            "marital": "married",
            "month": "may",
            "nr.employed": 5099.1,
            "pdays": 999,
            "poutcome": "failure",
            "previous": 1
          },
      ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())
```

This screenshot shows the response from the `POST` request to the endpoint and the payload used for testing the endpoint was saved to the `data.json` file:

![EndpointTst](assets/endpointtst.png)

After the endpoint test, a benchmarking test on the endpoint was conducted using the [Apache Benchmarking tool](https://httpd.apache.org/docs/2.4/programs/ab.html). The `benchmark.sh` contains the endpoint and authorization key as shown here:

```
ab -n 10 -v 4 -p data.json -T 'application/json' -H 'Authorization: Bearer gJaAltgcXjbyTpEDNsFd5Q1QFMXQMaQF' http://d8131649-149d-4a94-88d1-000024f32a51.southcentralus.azurecontainer.io/score
```

It sent the `data.json` file (from the endpoint test) to the endpoint 10 times and produced the run statistics as below:

![Benchmark](assets/banchmark.png)

The key takeaway from the benchmarking test is that there was no failed request among the 10 requests sent. The response time per request was `535.09` milliseconds which is well under the default timeout threshold of `60` seconds. Bear in mind the test was conducted in isolation from any interference, the response time measured in the real world environment may well be slower than this.

**5. Conclusion**

The _**Model Deployment**_ process was successfully executed producing a working best model endpoint, with a Swagger Documentation in tow and Application Insights enabled.

### Training Pipeline Automation

The process entails using `Python SDK` to publish the AutoML training pipeline, which can then be used to re-run the AutoML training pipeline on demand or schedule, thereby automating the AutoML training process. Additionally, publishing the pipeline enables a `REST` endpoint to rerun the pipeline from any HTTP library on any platform.

**1. Publish the AutoML training pipeline**

The pipeline used in AutoML Model training was published using `Python SDK`. This is the code snippet for publishing the pipeline:

```python
published_pipeline = pipeline_run.publish_pipeline(
    name="Bankmarketing Train", description="Training bankmarketing pipeline", version="1.0")

published_pipeline
```

The published pipeline object named `Bankmarketing Train` came with an endpoint and showed up on the _**Pipelines**_ console with an active status in the `AML Studio`, as shown here:

![PipelinePub](assets/pipelinePub.png)

The pipeline endpoint was viewable from the _**Pipelines**_ console by clicking the published pipeline name `Bankmarketing Train` to get to the _**Details**_ tab, like so:

![PipelineEP](assets/pipelineendptdtl.png)

**2. Post a request to the endpoint to start a run**

The next step was to send a `POST` request to the endpoint with an Experiment object named `pipeline-rest-endpoint` to trigger the pipeline run:

```python
import requests

rest_endpoint = published_pipeline.endpoint
response = requests.post(rest_endpoint,
                         headers=auth_header,
                         json={"ExperimentName": "pipeline-rest-endpoint"}
                        )
```

The experiment was visible on the _**Experiment**_ console in the `AML Studio`:

![PipelineExp](assets/pipelineexp.png)

Clicked on the Experiment `pipeline-rest-endpoint` link from the _**Experiments**_ console, a run `Run 1` was shown as running:

![PipelineExpDtl](assets/pipelineexpdtl.png)

Clicked on `Run 1` link from the _**Experiments**_ console to see the Pipeline run overview on the _**Graph**_ tab. It had an active status and a `HTTP` Run type, proving the run was triggered by the `POST` request to the published pipeline endpoint.

![PipelineExpDtlRun](assets/pipelinepexpdtlrun.png)

**3. Monitor the Pipeline run with Jupyter widget**

The Pipeline run was monitored from within the notebook with the RunDetails [Jupyter widget](http://jupyter.org/widgets). It was shown as running:

![PipelineEPRun](assets/pipelineeprun.png)

**4. Conclusion**

The _**Training Pipeline Automation**_ process was successfully implemented using `Python SDK`, with a published pipeline capable of accepting `HTTP` requests through its endpoint.

## Screen Recording

A screencast demonstraing the entire process of the working ML application, including interactions with the deployed model and published pipeline endpoints is available here:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=Mf03iVg8eZ8
" target="_blank"><img src="http://img.youtube.com/vi/Mf03iVg8eZ8/0.jpg"
alt="Operationalizing ML Project" width="240" height="180" border="10" /></a>

## Future Improvements

Suggested areas of improvement:
> * Apply model interpretability of AutoML on more complex and larger datasets, to gain speed and valuable insights in feature engineering, which can in turn be used to refine complex model accuracy
>
> * Add a CI/CD pipeline to interact with the Published Pipeline and trigger AutoML training run on scheduled or adhoc basis.
>
> * Apply the same concept learned here to create and publish other types of pipelines for
>
>   - Data Preparation
>   - Validation
>   - Deployment
>   - Combined tasks
>

## Citations

#### Project Starter Code
[Udacity Github Repo](https://github.com/udacity/nd00333_AZMLND_C2/tree/master/starter_files)

#### MLEMAND ND - Machine Learning Operations
[Lesson 2.5 - Exercise: Enable Security and Authentication](https://youtu.be/rsECJolX2Ns)

[Lesson 2.10 - Exercise: Deploy an Azure Machine learning Model](https://youtu.be/_RKfF1D6W24)

[Lesson 2.15 - Exercise: Enable Application Insights](https://youtu.be/EXGfNMMTuMY)

[Lesson 3.5 - Exercise: Swagger Documentation](https://youtu.be/3I-Oro-SWQs)

[Lesson 3.9 - Exercise: Consume Deployed Service](https://youtu.be/t4RYFKmdZ3Q)

[Lesson 3.13 - Exercise: Benchmark the Endpoint](https://youtu.be/z-kQdcGEUPQ)

[Lesson 4.5 - Exercise: Create a Pipeline](https://youtu.be/CV7bHfAyw8Y)

[Lesson 4.10 - Exercise: Publish and Consume a Pipeline](https://youtu.be/N007WceqyA0)
