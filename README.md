# NVIDIA Resource Suite

GPU job scheduling and resource management for the Sovereign Automation System.

## Role
Schedules GPU workloads across NVIDIA hardware for:
- TurboQuantum scoring engine
- NVQ mesh inference
- LLM fine-tuning jobs
- Vision agent processing

## Port: 7760

## API
POST /job/submit — Submit GPU job
GET /job/{id}/status — Poll job status  
GET /resources — List available GPU resources
GET /health — Health check

## SAP Integration
Node ID: `nvidia-resource-suite`
Managed by: DeepFlex Supervisor (port 8000)
