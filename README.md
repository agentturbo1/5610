# Fall 2018 5610 Term Project
*Authors: Tyler Thompson, Jeanette Arteaga, Lukas Gust*
## Information About the Files
#### Dependencies 
##### .dat
A .dat file containing the following contents in one line seperated by blank space:
1. beginning time
2. ending time
3. number of steps of trip
4. degrees latitude
5. minutes latitude
6. seconds latitude
7. hemisphere (1 if Northern, -1 if Southern)
8. degrees longitude
9. minutes longitude
10. seconds longitude
11. hemisphere (1 if Eastern, -1 if Western)
12. altitude of destination, in meters

There is also a data.dat file found here [data.dat](http://www.math.utah.edu/~pa/5610/tp/data.dat) that is needed to
initialize some constants.

##### vehicle.class
This simulation requries a java class file written by Peter Alfred found here:
[Peter Alfred Term Project Vehicle](http://www.math.utah.edu/~pa/5610/tp/vehicle.class)
This java class file depends on another class file called `angles.class`. This can be found here: 
[Peter Alfred Term Project Angles](http://www.math.utah.edu/~pa/5610/tp/angles.class)
This class file will take the data from the first .dat file mentioned above and generate a number of steps from a
certain point to the point described in the .dat file.

#### Python files
Both `satellite.py` and `receiver.py` were written by us, the authors of this README. They are needed to run the 
simulation each python script writes information to a log file. It also writes its intended output to standard output.
##### `satellite.py`
This script takes the positional output generated by `vehicle.class` and outputs signal times and cartesian coordinates
of the vehicle. This data is then sent to `receiver.py`.
##### `receiver.py`
This script takes input from the satellite and reconstructs the geographical coordinates. The coordinates are outputted 
to standard output.
## Running the Simulation
The simulation is easy to run. It requires java 1.8 and python 2.7.12. The system you run it on should support 
input/output piping. The following command template will work so long as you .dat file is correct.
```
cat <data_filename>.dat | java vehicle | python satellite.py | python receiver.py
```
This will output the steps of the trip according to the receiver. Compare this with the output of the vehicle to see 
how accurate the simulation was.