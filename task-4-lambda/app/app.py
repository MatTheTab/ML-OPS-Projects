#!/usr/bin/env python3
import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_lambda as _lambda,
    aws_s3_notifications as s3n,
    Duration,
    RemovalPolicy,
)
from constructs import Construct


class MlDeploymentStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # 1. Create S3 Bucket
        bucket = s3.Bucket(
            self,
            "InferenceBucket",
            removal_policy=RemovalPolicy.DESTROY,  # Only for homework!
            auto_delete_objects=True,
        )

        # 2. Create Lambda from Docker Image
        # CDK will automatically build the image and push it to ECR
        ml_lambda = _lambda.DockerImageFunction(
            self,
            "MLInferenceFunction",
            code=_lambda.DockerImageCode.from_image_asset("."),
            memory_size=2048,  # ML models need more RAM
            timeout=Duration.seconds(30),
        )

        # 3. Grant Permissions
        bucket.grant_read_write(ml_lambda)

        # 4. Set up Trigger
        bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.LambdaDestination(ml_lambda),
            s3.NotificationKeyFilter(suffix=".jpg"),  # Only trigger for JPGs
        )


app = cdk.App()
MlDeploymentStack(app, "MlDeploymentStack")
app.synth()
