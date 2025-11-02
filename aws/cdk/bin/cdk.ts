#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { ChronosTrainingStack } from '../lib/chronos-training-stack';

const app = new cdk.App();

// Get configuration from environment variables or CDK context
const environment = process.env.CDK_ENVIRONMENT || app.node.tryGetContext('environment') || 'dev';
const account = process.env.AWS_ACCOUNT_ID || process.env.CDK_DEFAULT_ACCOUNT;
const region = process.env.AWS_REGION || process.env.CDK_DEFAULT_REGION || 'us-east-1';
const projectName = process.env.PROJECT_NAME || app.node.tryGetContext('projectName') || 'Chronos-Training';

// Validate required configuration
if (!account) {
  throw new Error('AWS_ACCOUNT_ID must be set in .env file or environment. See aws/cdk/.env.example');
}

new ChronosTrainingStack(app, `${projectName}-${environment}-Stack`, {
  env: {
    account,
    region,
  },
  description: `Chronos Training Infrastructure (${environment}) - VPC, S3, IAM, Security Groups`,
});
