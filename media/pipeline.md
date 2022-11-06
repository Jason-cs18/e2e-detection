@startuml
title: Three-Stage Deployment
node Deep Learning Framework
node model_deployment
node model_benchmark


model_conversion -> model_deployment
model_deployment -> model_benchmark
@enduml