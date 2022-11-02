@startuml
actor user
node model_conversion
node model_deployment
node model_benchmark

user -[dashed]-> model_conversion: model\n& engine
user -[dashed]-> model_deployment: device\n(cpu/gpu)
model_conversion -> model_deployment
model_deployment -> model_benchmark
@enduml