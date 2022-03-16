
# Finding Nemo

## Installation

Alle Versionen unseres Templatematching können mit CMake bebaut werden und bekommen als
Kommandozeilenparameter jeweils 2 Argumente übergeben:
- den Dateipfad zum Bild, in dem das Template gesucht werden soll
- den Dateipfad zum Templatebild


### OpenMPI
Das OpenMPI Programm "template_matching_mpi.cpp" kann mit  
`mpiexec -n <worker count> <search area image path> <nemo template image path>`  
ausgeführt werden.


### OpenCL

Das OpenCL Programm "template_matching.cpp" kann mit CMale kompiliert werden. 
Zum Ausführen ist es jedoch wichtig, dass die Datei "kernel_source.cl" sich im gleichen Ordner wie das kompilierte, 
ausführbare Programm befindet. 
