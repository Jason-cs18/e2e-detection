@startuml
title: Video Analytics Pipeline
rectangle "Inference Engine" as engine{
rectangle "Default Model"
rectangle "Http"
}
actor admin
actor user

admin -[dashed]> engine: 1. select the \ndefault model\nconfig.py
user --> engine: 2. frame/video\nsend.py
engine --> user: 3. result\nxxx.py
user -[dashed]> user: 4. parse the result\nparse.py
@enduml