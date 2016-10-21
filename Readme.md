# Plot Google Map using Matplotlib

ported from Matlab version: https://www.mathworks.com/matlabcentral/fileexchange/27627-zoharby-plot-google-map

## Requirements and Dependencies
Only tested using python3.

- Pillow
- requests
- matplotlib
- numpy
- scipy

## Usage
Mostly the same as its Matlab version. Properties are passed in using keyword arguments.

## Limitation
`autoAxis` doesn't work, and should be left as `False`.

## Others
`getLatLonViaYql.py` is a little helper to get geo-location of cities from Yahoo Query. Also ported from its Matlab version.
