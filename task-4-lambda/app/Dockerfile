FROM public.ecr.aws/lambda/python:3.12

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Copy code and weights
# Ensure model.pth is in the same 'app' folder as this Dockerfile
COPY main.py ${LAMBDA_TASK_ROOT}
COPY model.pth ${LAMBDA_TASK_ROOT}

CMD [ "main.handler" ]