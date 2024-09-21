# LP02
A project to learn more about generative AI


### Running the code

#### Locally
To run the code locally, use a Python virtual environment.  You'll need AWS Credentials in your AWS Keyring, permissions to invoke Bedrock, and to have onboarded your account to use Claude 3.5 Sonnet.

```
# Start in the repo root

python3 -m venv venv
source venv/bin/activate

cd lp02
pipenv sync --dev
pipenv run streamlit run start_chat.py
```

This will launch a Streamlit app, which should redirect you to your default browser and allow you to chat w/ an AI Agent.

#### In AWS
The package uses Terraform to manage its cloud deployments.  You'll need valid AWS Credentials in your keyring (check using `aws sts get-caller-identity`).

```
cd lp02

./package.sh

terraform init
terraform plan
terraform apply
```

You can then run the Lambda manually in the AWS console using test events.

### Dependencies
`pipenv` is used to managed dependencies within the project.  The `Pipefile` and `Pipefile.lock` handle the local environment.  You can add dependencies like so:

```
pipenv install boto3
```

This updates the `Pipfile`/`Pipfile.lock` with the new dependency.  To create a local copy of the dependencies, such as for bundling a distribution, you can use pip like so:

```
pipenv requirements > requirements.txt
python3 -m pip install -r requirements.txt -t ./package --upgrade

zip -r9 lp02.zip tools/ package/
```