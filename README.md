# otmarkov
OpenTURNS experiments with markov chains

TODO-List:
* The current implementation does not manage a multidimensionnal state variable. 
Moreover, the current implementation does not allow to access to the full chain. 
Create a Process where the dimension of the mesh is 1D: this is the "time". 
The dimension of the process is the dimension of the state.
This way, one can generate such a process and get the full chain of 
states. 
