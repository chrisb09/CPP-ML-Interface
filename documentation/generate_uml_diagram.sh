#!/bin/bash


puml_path="new.puml"
puml_cmd="plantuml"
plantuml_jar="$HOME/scripts/plantuml-SNAPSHOT.jar"

# Check if a parameter is provided to override the default puml file path
if [ $# -eq 1 ]; then
    puml_path="$1"
fi

echo "Using PUML file: $puml_path"

# Check if PUML file exists
if [ ! -f "$puml_path" ]; then
    echo "PUML file '$puml_path' not found!"
    exit 1
fi

# check if user is ro092286

if [ "$(whoami)" == "ro092286" ]; then

    puml_cmd="java -jar $plantuml_jar"
    java_module="Java/21.0.5"

    module_is_loaded=0

    module is-loaded $java_module && {
        module_is_loaded=1
    }

    if [ "$module_is_loaded" -eq 1 ]; then
        echo "$java_module is loaded!"
    else
        echo "$java_module not loaded. Loading it now..."
        module load $java_module || { echo "Failed to load $java_module"; exit 1; }
    fi
fi

echo "Generating UML diagrams from $puml_path using command: $puml_cmd"

if [[ "$puml_cmd" == java* ]] && [ ! -f "$plantuml_jar" ]; then
    if command -v plantuml >/dev/null 2>&1; then
        echo "Jar not found at $plantuml_jar, falling back to system plantuml"
        puml_cmd="plantuml"
    else
        echo "PlantUML jar not found at $plantuml_jar and system 'plantuml' is unavailable."
        exit 1
    fi
fi

$puml_cmd "$puml_path" -tpng
$puml_cmd "$puml_path" -tsvg