@startuml
actor user
node model_conversion
node model_deployment
node model_benchmark
file report
note right of report
benchmark setup
device info
model info
testing data
output and accuracy
resource usage
end note

user -[dashed]-> model_conversion: model\n& engine
user -[dashed]-> model_deployment: device\n(cpu/gpu)
model_conversion -> model_deployment
model_deployment -> model_benchmark
model_benchmark -> report
@enduml